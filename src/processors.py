"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
from .compute_metrics import superglue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class OurInputExample(InputExample):
    """
    Inherit from Transformers' InputExample.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    text_c: Optional[str] = None
    text_d: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched"
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched",
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test_mismatched",
        )


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        text_index = 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 3
        q2_index = 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(
                os.path.join(data_dir, "train.csv"), header=None
            ).values.tolist(),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(),
            "dev",
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(
                os.path.join(data_dir, "test.csv"), header=None
            ).values.tolist(),
            "test",
        )

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=line[1] + ". " + line[2],
                        short_text=line[1] + ".",
                        label=line[0],
                    )
                )
            elif self.task_name == "yelp_review_full":
                examples.append(
                    InputExample(
                        guid=guid, text_a=line[1], short_text=line[1], label=line[0]
                    )
                )
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += " " + line[2]
                if not pd.isna(line[3]):
                    text += " " + line[3]
                examples.append(
                    InputExample(
                        guid=guid, text_a=text, short_text=line[1], label=line[0]
                    )
                )
            elif self.task_name in ["mr", "sst-5", "subj", "trec", "cr", "mpqa"]:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples


class BoolqProcessor(DataProcessor):
    """Processor for the BoolQ data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["passage"]
            text_b = line["question"]
            label = line["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class CbProcessor(DataProcessor):
    """Processor for the CommitmentBank data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["entailment", "contradiction", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dwicict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, manual=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            "train",
            manual=manual,
        )

    def get_dev_examples(self, data_dir, manual=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            "dev",
            manual=manual,
        )

    def get_test_examples(self, data_dir, manual=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            "test",
            manual=manual,
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, manual=False):
        """Creates examples for the training and dev sets."""
        examples = []
        labels_mapping = {}
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            label = line["label"]
            premise = line["premise"]
            choice1 = line["choice1"]
            choice2 = line["choice2"]
            question = line["question"]
            if manual:
                if question == "effect":
                    question = "so"
                else:
                    question = "because"
                labels_mapping[guid] = [choice1[:-1], choice2[:-1]]
            examples.append(
                OurInputExample(
                    guid=guid,
                    text_a=choice1,
                    text_b=choice2,
                    text_c=premise,
                    text_d=question,
                    label=label,
                )
            )
        if manual:
            return examples, labels_mapping
        else:
            return examples


class MultircProcessor(DataProcessor):
    """Processor for the Multirc data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # NOTE(Alex): currently concatenating passage and question,
        # which might lead to the question getting cut off. An
        # alternative is to concatenate question and answer, but that
        # feels like there's a missing [SEP] token. Maybe the robust
        # solution is to use [SEP] tokens between everything.
        examples = []
        for (i, line) in enumerate(lines):
            guid = [line["tid"], line["qid"], line["aid"]]
            label = line["label"]
            text = line["text"]
            question = line["question"]
            answer = line["answer"]
            examples.append(
                OurInputExample(
                    guid=guid,
                    text_a=text,
                    text_b=question,
                    text_c=answer,
                    text_d=None,
                    label=label,
                )
            )
        return examples


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set (SuperGLUE version)."""

    def __init__(self):
        self._answers = None

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, fine_tuning=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            "train",
            fine_tuning=fine_tuning,
        )

    def get_dev_examples(self, data_dir, fine_tuning=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            "dev",
            fine_tuning=fine_tuning,
        )

    def get_test_examples(self, data_dir, fine_tuning=False):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            "test",
            fine_tuning=fine_tuning,
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, fine_tuning=False):
        """Creates examples for the training and dev sets."""
        examples = []
        qst2ans = {}
        for (i, line) in enumerate(lines):
            text_a = line["text"]
            text_b = line["entities"]
            text_c = line["query"]
            if not fine_tuning:
                text_c = text_c.replace("@placeholder", "<extra_id_0>")
            if set_type == "train":
                for i, answer in enumerate(line["answers"]):
                    guid = [line["tid"], line["qid"], i]
                    label = answer["text"]
                    qst2ans[tuple(guid)] = label
                    examples.append(
                        OurInputExample(
                            guid=guid,
                            text_a=text_a,
                            text_b=text_b,
                            text_c=text_c,
                            label=label,
                        )
                    )
            else:
                guid = [line["tid"], line["qid"]]
                label = [answer["text"] for answer in line["answers"]]
                qst2ans[tuple(guid)] = label
                examples.append(
                    OurInputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=text_b,
                        text_c=text_c,
                        label=label,
                    )
                )
        return examples, qst2ans


class WicProcessor(DataProcessor):
    """Processor for the WiC data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            label = line["label"]
            sentence1 = line["sentence1"]
            sentence2 = line["sentence2"]
            word = line["word"]
            text_a = f"sentence1: {sentence1} sentence2: {sentence2} word: {word}"
            examples.append(
                OurInputExample(
                    guid=guid,
                    text_a=sentence1,
                    text_b=sentence2,
                    text_c=word,
                    text_d=None,
                    label=label,
                )
            )
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set (SuperGLUE version)."""

    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return [json.loads(l) for l in f]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test"
        )

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            if set_type == "train" and line["label"] == False:
                continue
            guid = line["idx"]
            text_a = line["text"]
            text_b = line["target"]["span1_text"]
            text_c = "'*" + line["target"]["span2_text"] + "*'"
            label = line["label"]
            examples.append(
                OurInputExample(
                    guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label
                )
            )
        return examples


def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}


# Add your task to the following mappings

processors_mapping = {
    "cola": ColaProcessor(),
    "mnli": MnliProcessor(),
    "mnli-mm": MnliMismatchedProcessor(),
    "mrpc": MrpcProcessor(),
    "sst-2": Sst2Processor(),
    "sts-b": StsbProcessor(),
    "qqp": QqpProcessor(),
    "qnli": QnliProcessor(),
    "rte": RteProcessor(),
    "wnli": WnliProcessor(),
    "snli": SnliProcessor(),
    "mr": TextClassificationProcessor("mr"),
    "sst-5": TextClassificationProcessor("sst-5"),
    "subj": TextClassificationProcessor("subj"),
    "trec": TextClassificationProcessor("trec"),
    "cr": TextClassificationProcessor("cr"),
    "mpqa": TextClassificationProcessor("mpqa"),
    "boolq": BoolqProcessor(),
    "cb": CbProcessor(),
    "copa": CopaProcessor(),
    "multirc": MultircProcessor(),
    "record": RecordProcessor(),
    "wic": WicProcessor(),
    "wsc": WscProcessor(),
}

num_labels_mapping = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "mr": 2,
    "sst-5": 5,
    "subj": 2,
    "trec": 6,
    "cr": 2,
    "mpqa": 2,
    "boolq": 2,
    "cb": 3,
    "copa": 2,
    "multirc": 2,
    "record": 2,
    "wic": 2,
    "wsc": 2,
}

superglue_tasks_num_spans = {
    "wic": 2,
    "wsc": 2,
}

output_modes_mapping = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "snli": "classification",
    "mr": "classification",
    "sst-5": "classification",
    "subj": "classification",
    "trec": "classification",
    "cr": "classification",
    "mpqa": "classification",
    "boolq": "classification",
    "cb": "classification",
    "copa": "classification",
    "multirc": "classification",
    "record": "classification",
    "wic": "classification",
    "wsc": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "cola": glue_compute_metrics,
    "mnli": glue_compute_metrics,
    "mnli-mm": glue_compute_metrics,
    "mrpc": glue_compute_metrics,
    "sst-2": glue_compute_metrics,
    "sts-b": glue_compute_metrics,
    "qqp": glue_compute_metrics,
    "qnli": glue_compute_metrics,
    "rte": glue_compute_metrics,
    "wnli": glue_compute_metrics,
    "snli": text_classification_metrics,
    "mr": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
    "cr": text_classification_metrics,
    "mpqa": text_classification_metrics,
    "boolq": superglue_compute_metrics,
    "cb": superglue_compute_metrics,
    "copa": superglue_compute_metrics,
    "multirc": superglue_compute_metrics,
    "record": superglue_compute_metrics,
    "wic": superglue_compute_metrics,
    "wsc": superglue_compute_metrics,
}

# For regression task only: median
median_mapping = {"sts-b": 2.5}

bound_mapping = {"sts-b": (0, 5)}
