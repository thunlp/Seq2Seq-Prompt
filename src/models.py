"""Custom models for few-shot learning specific operations."""

import re
import string
from collections import Counter
import logging
import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaModel,
    T5ForConditionalGeneration,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaLMHead,
)


logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, "bert"):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(
        new_num_types, old_token_type_embeddings.weight.size(1)
    )
    if not random_segment:
        new_token_type_embeddings.weight.data[
            : old_token_type_embeddings.weight.size(0)
        ] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, "bert"):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        guid=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)  # newly
        self.lm_head = RobertaLMHead(config)  # newly
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_words_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        guid=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_words_list)):
            logits.append(
                prediction_mask_scores[:, self.label_words_list[label_id][0]].unsqueeze(
                    -1
                )
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                # labels contain label id e.g [0,1]
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


"""
AutoSeq
"""


class T5ForPromptFinetuning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_words_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For COPA
        self.copa_train_mapping = None
        self.copa_eval_mapping = None
        self.copa_test_mapping = None
        self.copa_mode = None

        # For ReCoRD
        self.record_train_mapping = None
        self.record_eval_mapping = None
        self.record_test_mapping = None
        self.record_mode = None

        # For WSC
        self.wsc_train_mapping = None
        self.wsc_eval_mapping = None
        self.wsc_test_mapping = None
        self.wsc_mode = None

        # For label search.
        self.return_full_softmax = None

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script"""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        """Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script"""
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _record_f1_score(self, prediction, ground_truth):
        """Compute normalized token level F1
        From official ReCoRD eval script"""
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _record_em_score(self, prediction, ground_truth):
        """Compute normalized exact match
        From official ReCoRD eval script"""
        return self._normalize_answer(prediction) == self._normalize_answer(
            ground_truth
        )

    def _wsc_simple(self, prediction, ground_truth):
        """Sees whether we predicted the referent or not."""
        determiners = {
            "a",
            "an",
            "few",
            "her",
            "his",
            "each",
            "every",
            "many",
            "much",
            "my",
            "our",
            "some",
            "that",
            "the",
            "their",
            "these",
            "this",
            "those",
            "which",
            "whose",
            "your",
        }

        def clean(s):
            """Ignore capitalization and determiners."""
            s = s.strip().lower()
            return " ".join([w for w in s.split(" ") if w not in determiners])

        # We aren't using the label but rather using the extracted referent so that we
        # can see if the prediction is equivalent to the referent.
        prediction = clean(prediction)
        referent = clean(ground_truth)

        if ("'" in prediction) != ("'" in referent):
            # Make sure we don't mark cases where the prediction is "Bob" and the
            # referent is "Bob's hat" as predicting the referent.
            predicted_referent = False
        else:
            prediction_words = set(prediction.split(" "))
            referent_words = set(referent.split(" "))

            # Handle cases where the prediction is "fuzzy bunny" and the referent is
            # "bunny".
            predicted_referent = prediction_words.issubset(
                referent_words
            ) or referent_words.issubset(prediction_words)

        return int(predicted_referent)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        guid=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        if self.config.finetuning_task == "copa" and "man" in self.data_args.tag:
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.copa_train_mapping[guid[i].item()][
                                    labels[i].item()
                                ]
                            ).input_ids
                        )
                        for i in range(len(guid))
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, id in enumerate(guid):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.copa_train_mapping[id.item()][labels[i].item()]
                            ).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(
                            "<extra_id_0> "
                            + self.copa_train_mapping[id.item()][labels[i].item()]
                        ).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            # Softmax loss
            predictions = torch.zeros((2, batch_size), requires_grad=True).cuda()
            for i in range(batch_size):
                for label_id in [0, 1]:
                    if self.copa_mode == "train":
                        example_labels = torch.tensor(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.copa_eval_mapping[guid[i].item()][label_id]
                            ).input_ids
                        )
                    else:
                        example_labels = torch.tensor(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.copa_test_mapping[guid[i].item()][label_id]
                            ).input_ids
                        )
                    outputs = self.t5(
                        input_ids=input_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        labels=example_labels.unsqueeze(0).cuda(),
                        return_dict=True,
                    )
                    logsoftmax = nn.LogSoftmax(-1)
                    prediction_mask_scores = logsoftmax(
                        outputs.logits
                    )  # Shape (batch_size, sequence_length, config.vocab_size)
                    seq_len = example_labels.size(0)
                    for j in range(seq_len):
                        predictions[label_id][i] += prediction_mask_scores[0][j][
                            example_labels[j].item()
                        ]  # after log
            # Return logits of for each label
            logits = torch.cat(
                tuple([predictions[i].unsqueeze(-1) for i in [0, 1]]),
                -1,
            )

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)).cuda(), labels.view(-1))

            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        if self.config.finetuning_task == "record":
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.record_train_mapping[
                                    tuple(id.cpu().numpy().tolist())
                                ]
                            ).input_ids
                        )
                        for id in guid
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, id in enumerate(guid):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(
                                "<extra_id_0> "
                                + self.record_train_mapping[
                                    tuple(id.cpu().numpy().tolist())
                                ]
                            ).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(
                            "<extra_id_0> "
                            + self.record_train_mapping[
                                tuple(id.cpu().numpy().tolist())
                            ]
                        ).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
            # Return logits of for each label
            logits = torch.zeros((batch_size, 2))
            for i, output in enumerate(outputs):
                pred = (
                    self.tokenizer.decode(output.cpu())
                    .replace("<pad>", "")
                    .replace("</s>", "")
                    .replace("<extra_id_0>", "")
                    .replace("<extra_id_1>", "")
                )
                golds = []
                if self.record_mode == "train":
                    golds = self.record_eval_mapping[
                        tuple(guid[i].cpu().numpy().tolist())
                    ]
                else:
                    golds = self.record_test_mapping[
                        tuple(guid[i].cpu().numpy().tolist())
                    ]
                logits[i][0] = self._metric_max_over_ground_truths(
                    self._record_f1_score, pred, golds
                )
                logits[i][1] = self._metric_max_over_ground_truths(
                    self._record_em_score, pred, golds
                )

            output = (logits,)
            return (torch.tensor(0.0),) + output

        if self.config.finetuning_task == "wsc":
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(
                            self.tokenizer(
                                "<extra_id_0> " + self.wsc_train_mapping[id.item()]
                            ).input_ids
                        )
                        for id in guid
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, id in enumerate(guid):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(
                                "<extra_id_0> " + self.wsc_train_mapping[id.item()]
                            ).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(
                            "<extra_id_0> " + self.wsc_train_mapping[id.item()]
                        ).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
            # Return logits of for each label
            logits = torch.zeros((batch_size, self.num_labels))
            for i, output in enumerate(outputs):
                string1 = (
                    self.tokenizer.decode(output.cpu())
                    .replace("<pad>", "")
                    .replace("</s>", "")
                    .replace("<extra_id_0>", "")
                    .replace("<extra_id_1>", "")
                )
                string2 = ""
                if self.wsc_mode == "train":
                    string2 = self.wsc_eval_mapping[guid[i].item()]
                else:
                    string2 = self.wsc_test_mapping[guid[i].item()]
                if self._wsc_simple(string1, string2):
                    logits[i][0] = 1
                else:
                    logits[i][0] = -1

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)).cuda(), labels.view(-1))

            output = (logits,)
            return (loss,) + output

        # Language modeling loss
        if self.config.num_labels != 1 and getattr(self, "training") == True:
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            max_label_len = max([len(i) for i in self.label_words_list]) + 2
            example_labels = (
                -100 * torch.ones(batch_size, max_label_len).long()
            )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
            for i, label in enumerate(labels):
                example_label = [32099] + self.label_words_list[label.item()] + [1]
                example_labels[i][0 : len(example_label)] = torch.tensor(example_label)
            outputs = self.t5(
                input_ids,
                attention_mask=attention_mask,
                labels=example_labels.cuda(),
            )
            return outputs

        # Softmax loss
        predictions = (
            torch.zeros((self.num_labels, batch_size), requires_grad=True).cuda()
            if self.config.num_labels != 1
            else torch.zeros((2, batch_size), requires_grad=True).cuda()
        )
        for label_id in range(len(self.label_words_list)):
            example_labels = torch.tensor(
                [[32099] + self.label_words_list[label_id] + [1] for _ in labels]
            )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=example_labels.cuda(),
                return_dict=True,
            )
            logsoftmax = nn.LogSoftmax(-1)
            prediction_mask_scores = logsoftmax(
                outputs.logits
            )  # Shape (batch_size, sequence_length, config.vocab_size)
            seq_len = example_labels.size(1)
            for i in range(batch_size):
                for j in range(seq_len):
                    predictions[label_id][i] += prediction_mask_scores[i][j][
                        example_labels[i][j].item()
                    ]  # after log
        # Return logits of for each label
        logits = torch.cat(
            tuple(
                [
                    predictions[i].unsqueeze(-1)
                    for i in range(
                        self.num_labels if self.config.num_labels != 1 else 2
                    )
                ]
            ),
            -1,
        )

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)).cuda(), labels.view(-1)
                )

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


"""
Fine-tuning
"""


class T5ForFinetuning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_words_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For ReCoRD
        self.record_train_mapping = None
        self.record_eval_mapping = None
        self.record_test_mapping = None
        self.record_mode = None

        # For WSC
        self.wsc_train_mapping = None
        self.wsc_eval_mapping = None
        self.wsc_test_mapping = None
        self.wsc_mode = None

        # For label search.
        self.return_full_softmax = None

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script"""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        """Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script"""
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _record_f1_score(self, prediction, ground_truth):
        """Compute normalized token level F1
        From official ReCoRD eval script"""
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _record_em_score(self, prediction, ground_truth):
        """Compute normalized exact match
        From official ReCoRD eval script"""
        return self._normalize_answer(prediction) == self._normalize_answer(
            ground_truth
        )

    def _wsc_simple(self, prediction, ground_truth):
        """Sees whether we predicted the referent or not."""
        determiners = {
            "a",
            "an",
            "few",
            "her",
            "his",
            "each",
            "every",
            "many",
            "much",
            "my",
            "our",
            "some",
            "that",
            "the",
            "their",
            "these",
            "this",
            "those",
            "which",
            "whose",
            "your",
        }

        def clean(s):
            """Ignore capitalization and determiners."""
            s = s.strip().lower()
            return " ".join([w for w in s.split(" ") if w not in determiners])

        # We aren't using the label but rather using the extracted referent so that we
        # can see if the prediction is equivalent to the referent.
        prediction = clean(prediction)
        referent = clean(ground_truth)

        if ("'" in prediction) != ("'" in referent):
            # Make sure we don't mark cases where the prediction is "Bob" and the
            # referent is "Bob's hat" as predicting the referent.
            predicted_referent = False
        else:
            prediction_words = set(prediction.split(" "))
            referent_words = set(referent.split(" "))

            # Handle cases where the prediction is "fuzzy bunny" and the referent is
            # "bunny".
            predicted_referent = prediction_words.issubset(
                referent_words
            ) or referent_words.issubset(prediction_words)

        return int(predicted_referent)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        guid=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        if self.config.num_labels == 1:
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(self.tokenizer(str(round(label.item() * 5) / 5)).input_ids)
                        for label in labels
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, label in enumerate(labels):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(str(round(label.item() * 5) / 5)).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(str(round(label.item() * 5) / 5)).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
            # Return logits of for each label
            logits = torch.zeros((batch_size, 1))
            for i, output in enumerate(outputs):
                string = (
                    self.tokenizer.decode(output.cpu())
                    .replace("<pad>", "")
                    .replace("</s>", "")
                )
                try:
                    logits[i][0] = float(string)
                except:
                    logits[i][0] = -1.0

            output = (logits,)
            return (torch.tensor(0.0),) + output

        if self.config.finetuning_task == "record":
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(
                            self.tokenizer(
                                self.record_train_mapping[
                                    tuple(id.cpu().numpy().tolist())
                                ]
                            ).input_ids
                        )
                        for id in guid
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, id in enumerate(guid):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(
                                self.record_train_mapping[
                                    tuple(id.cpu().numpy().tolist())
                                ]
                            ).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(
                            self.record_train_mapping[tuple(id.cpu().numpy().tolist())]
                        ).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
            # Return logits of for each label
            logits = torch.zeros((batch_size, 2))
            for i, output in enumerate(outputs):
                pred = (
                    self.tokenizer.decode(output.cpu())
                    .replace("<pad>", "")
                    .replace("</s>", "")
                )
                golds = []
                if self.record_mode == "train":
                    golds = self.record_eval_mapping[
                        tuple(guid[i].cpu().numpy().tolist())
                    ]
                else:
                    golds = self.record_test_mapping[
                        tuple(guid[i].cpu().numpy().tolist())
                    ]
                logits[i][0] = self._metric_max_over_ground_truths(
                    self._record_f1_score, pred, golds
                )
                logits[i][1] = self._metric_max_over_ground_truths(
                    self._record_em_score, pred, golds
                )

            output = (logits,)
            return (torch.tensor(0.0),) + output

        if self.config.finetuning_task == "wsc":
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            if getattr(self, "training") == True:
                max_label_len = max(
                    [
                        len(self.tokenizer(self.wsc_train_mapping[id.item()]).input_ids)
                        for id in guid
                    ]
                )
                example_labels = (
                    -100 * torch.ones(batch_size, max_label_len).long()
                )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
                for i, id in enumerate(guid):
                    example_labels[i][
                        0 : len(
                            self.tokenizer(self.wsc_train_mapping[id.item()]).input_ids
                        )
                    ] = torch.tensor(
                        self.tokenizer(self.wsc_train_mapping[id.item()]).input_ids
                    )
                outputs = self.t5(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=example_labels.cuda(),
                )
                return outputs

            outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
            # Return logits of for each label
            logits = torch.zeros((batch_size, self.num_labels))
            for i, output in enumerate(outputs):
                string1 = (
                    self.tokenizer.decode(output.cpu())
                    .replace("<pad>", "")
                    .replace("</s>", "")
                )
                string2 = ""
                if self.wsc_mode == "train":
                    string2 = self.wsc_eval_mapping[guid[i].item()]
                else:
                    string2 = self.wsc_test_mapping[guid[i].item()]
                if self._wsc_simple(string1, string2):
                    logits[i][0] = 1
                else:
                    logits[i][0] = -1

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)).cuda(), labels.view(-1))

            output = (logits,)
            return (loss,) + output

        # Language modeling loss
        if getattr(self, "training") == True:
            # In training, return all logits for backward. Only logits for the label_word changes the performance slightly
            max_label_len = max([len(i) for i in self.label_words_list]) + 1
            example_labels = (
                -100 * torch.ones(batch_size, max_label_len).long()
            )  # Generate labels of shape (batch_size, target_sequence_length). The input only has one <extra_id_*>. The tokenizer automatically adds "</s>" token
            for i, label in enumerate(labels):
                example_labels[i][
                    0 : len(self.label_words_list[label.item()]) + 1
                ] = torch.tensor(self.label_words_list[label.item()] + [1])
            outputs = self.t5(
                input_ids,
                attention_mask=attention_mask,
                labels=example_labels.cuda(),
            )
            return outputs

        outputs = self.t5.generate(input_ids, attention_mask=attention_mask)
        # Return logits of for each label
        logits = torch.zeros((batch_size, self.num_labels))
        for i, label in enumerate(labels):
            if outputs[i].size(0) < len(self.label_words_list[label.item()]) + 2:
                logits[i][label.item()] = -1
                continue
            processed_label = torch.zeros(outputs[i].size(0)).long()
            processed_label[
                1 : len(self.label_words_list[label.item()]) + 2
            ] = torch.tensor(self.label_words_list[label.item()] + [1])
            if outputs[i].cpu().equal(processed_label):
                logits[i][label.item()] = 1  # correct
            else:
                logits[i][label.item()] = -1  # wrong

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)).cuda(), labels.view(-1))

        output = (logits,)
        return (loss,) + output
