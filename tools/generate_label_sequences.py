from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd
import copy
import jsonlines
import math


def get_text(template, input_text_tuple, label, tokenizer, mapping):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    special_token_mapping = {
        "cls": tokenizer.cls_token_id,
        "mask": tokenizer.mask_token_id,
        "sep": tokenizer.sep_token_id,
        "sep+": tokenizer.sep_token_id,
    }
    for i in range(10):
        special_token_mapping["<extra_id_%d>" % (i)] = tokenizer.convert_tokens_to_ids(
            "<extra_id_%d>" % (i)
        )
    template_list = template.split("*")
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == "cls" and "T5" in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:5] == "label":
            new_tokens += enc(" " + mapping[label])
        elif part[:5] == "sent_":
            sent_id = int(part.split("_")[1])
            new_tokens += enc(input_text_tuple[sent_id])
        elif part[:6] == "+sent_":
            sent_id = int(part.split("_")[1])
            new_tokens += enc(" " + input_text_tuple[sent_id])  # add space
        elif part[:6] == "sent-_":
            # Delete the last token
            sent_id = int(part.split("_")[1])
            new_tokens += enc(input_text_tuple[sent_id][:-1])
        elif part[:7] == "+sentl_":
            # Lower case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(" " + text)
        elif part[:7] == "+sentu_":
            # Upper case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(" " + text)
        elif part[:6] == "sentl_":
            # Lower case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:6] == "sentu_":
            # Lower case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == "sentl-_":
            # Lower case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        else:
            part = part.replace(
                "_", " "
            )  # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer.convert_tokens_to_ids(part))
            else:
                new_tokens += enc(part)

        input_ids += new_tokens
    return input_ids


def generate(
    dataset,
    template,
    model,
    tokenizer,
    target_number,
    mapping,
    beam,
    content_length,
    label=None,
    length_limit=None,
    truncate=None,
):
    """
    Generate templates based on given inputs

    label: Only use instances with this label
    length_limit: At least generate content as long as length_limit (deprecated)
    """
    input_tensors = []
    max_length = 0

    # Process the inputs
    for item in dataset:
        if item["label"] == label:
            input_text = get_text(
                template, item["text"], item["label"], tokenizer, mapping
            )
            if truncate is not None:
                if truncate == "head":
                    input_text = input_text[-256:]
                elif truncate == "tail":
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, : input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, : input_tensors[i].size(-1)] = 1

    # Print some examples
    print("####### example #######")
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))
    print(tokenizer.decode(input_ids[2]))
    print("####### example #######\n")

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20

    start_mask = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [
        {
            "decoder_input_ids": ori_decoder_input_ids,
            "ll": 0,
            "output_id": 1,
            "output": [],
            "last_length": -1,
        }
    ]
    for i in tqdm(range(max_length - 2)):
        new_current_output = []
        # for every pattern
        for item in current_output:
            if item["output_id"] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item["decoder_input_ids"]

            # Forward
            batch_size = 16
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(
                        model(
                            input_ids[start:end],
                            attention_mask=attention_mask[start:end],
                            decoder_input_ids=decoder_input_ids.cuda()[start:end],
                        )[0]
                    )
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], -1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[: beam + 3]

            for word_id in ids:
                output_id = item["output_id"]

                if (
                    word_id == start_mask - output_id
                    or word_id == tokenizer.convert_tokens_to_ids("</s>")
                ):
                    # Finish one part
                    if (
                        length_limit is not None
                        and item["last_length"] < length_limit[output_id - 1]
                    ):
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item["last_length"] + 1
                    check = True

                output_text = item["output"] + [word_id]
                ll = item["ll"] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(
                    decoder_input_ids.size()
                )
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if (
                    word_id
                    in [
                        2,
                        3,
                        96,
                        137,
                        233,
                        535,
                        1141,
                        1280,
                        1820,
                        2824,
                        4275,
                        4609,
                        4720,
                        7067,
                        9374,
                        10011,
                        14125,
                        16463,
                        19794,
                        22354,
                    ]
                    or (i == 1 and word_id == 1)
                ):
                    check = False

                # Forbid continuous "."
                if (
                    len(output_text) > 1
                    and output_text[-2]
                    in [
                        5,
                        6,
                        10,
                        18,
                        31,
                        41,
                        55,
                        58,
                        61,
                        105,
                        117,
                        121,
                        153,
                        1239,
                        1603,
                        3155,
                        3158,
                        4697,
                        4819,
                        6810,
                        8546,
                        8665,
                        10769,
                        12887,
                        20462,
                    ]
                    and output_text[-1]
                    in [
                        5,
                        6,
                        10,
                        18,
                        31,
                        41,
                        55,
                        58,
                        61,
                        105,
                        117,
                        121,
                        153,
                        1239,
                        1603,
                        3155,
                        3158,
                        4697,
                        4819,
                        6810,
                        8546,
                        8665,
                        10769,
                        12887,
                        20462,
                    ]
                ):
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {
                        "decoder_input_ids": new_decoder_input_ids,
                        "ll": ll,
                        "output_id": output_id,
                        "output": output_text,
                        "last_length": last_length,
                    }
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x["ll"], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    result = []
    print("####### generated results #######")
    for item in current_output:
        generate_text = ""
        for token in item["output"]:
            generate_text += tokenizer.convert_ids_to_tokens(token)
        print("--------------")
        print("score:", math.exp(item["ll"].item()))
        print("generated ids", item["output"])
        print("generated text", generate_text)
        result.append([generate_text, math.exp(item["ll"].item())])
    print("####### generated results #######\n")

    return result


def load_dataset(task, data_dir):
    if task in [
        "MNLI",
        "MRPC",
        "QNLI",
        "QQP",
        "RTE",
        "SNLI",
        "SST-2",
        "STS-B",
        "WNLI",
        "CoLA",
    ]:
        lines = open(os.path.join(data_dir, "train.tsv")).readlines()
        if task != "CoLA":
            lines = lines[1:]

        dataset = []
        for line in lines:
            line = line.strip().split("\t")
            if task == "CoLA":
                dataset.append({"label": line[1], "text": [line[-1]]})
            elif task == "MNLI":
                dataset.append({"label": line[-1], "text": [line[8], line[9]]})
            elif task == "MRPC":
                dataset.append({"label": line[0], "text": [line[-2], line[-1]]})
            elif task == "QNLI":
                dataset.append({"label": line[-1], "text": [line[1], line[2]]})
            elif task == "QQP":
                dataset.append({"label": line[-1], "text": [line[3], line[4]]})
            elif task == "RTE":
                dataset.append({"label": line[-1], "text": [line[1], line[2]]})
            elif task == "SNLI":
                dataset.append({"label": line[-1], "text": [line[7], line[8]]})
            elif task == "SST-2":
                dataset.append({"label": line[-1], "text": [line[0]]})
            elif task == "STS-B":
                dataset.append(
                    {
                        "label": "0" if float(line[-1]) < 2.5 else "1",
                        "text": [line[-3], line[-2]],
                    }
                )
            elif task == "WNLI":
                dataset.append({"label": line[-1], "text": [line[1], line[2]]})
            else:
                raise NotImplementedError
    elif task in [
        "AX-b",
        "AX-g",
        "BoolQ",
        "CB",
        "COPA",
        "MultiRC",
        "WiC",
    ]:
        dataset = []
        for line in jsonlines.Reader(
            open(os.path.join(data_dir, "train.jsonl"), "r+", encoding="utf8")
        ):
            if task == "AX-b":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["sentence1"], line["sentence2"]],
                    }
                )
            elif task == "AX-g":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["hypothesis"], line["premise"]],
                    }
                )
            elif task == "BoolQ":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["question"], line["passage"]],
                    }
                )
            elif task == "CB":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["premise"], line["hypothesis"]],
                    }
                )
            elif task == "COPA":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [
                            line["premise"],
                            line["question"],
                            line["choice1"],
                            line["choice2"],
                        ],
                    }
                )
            elif task == "MultiRC":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["text"], line["question"], line["answer"]],
                    }
                )
            elif task == "WiC":
                dataset.append(
                    {
                        "label": line["label"],
                        "text": [line["sentence1"], line["sentence2"], line["word"]],
                    }
                )
    else:
        lines = pd.read_csv(
            os.path.join(data_dir, "train.csv"), header=None
        ).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({"label": line[0], "text": [line[1]]})

    return dataset


def get_prob(
    dataset,
    template,
    model,
    tokenizer,
    mapping,
    decoder_input,
    label=None,
    truncate=None,
):
    input_tensors = []
    max_length = 0

    # Process the inputs
    for item in dataset:
        if item["label"] == label:
            input_text = get_text(
                template, item["text"], item["label"], tokenizer, mapping
            )
            if truncate is not None:
                if truncate == "head":
                    input_text = input_text[-256:]
                elif truncate == "tail":
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, : input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, : input_tensors[i].size(-1)] = 1
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20

    example_labels = torch.zeros((input_ids.size(0), max_length)).long()
    example_label = tokenizer("<extra_id_0> " + decoder_input).input_ids
    example_labels[..., 0 : len(example_label)] = torch.tensor(example_label)

    with torch.no_grad():
        aggr_output = model(
            input_ids,
            attention_mask=attention_mask,
            labels=example_labels.cuda(),
        )[1]
        aggr_output = aggr_output.mean(0)  # Shape (sequence_length, config.vocab_size)
        ll = 0
        for i, word_id in enumerate(example_label):
            ll += (
                aggr_output[i][word_id].item()
                - torch.logsumexp(aggr_output[i], -1).item()
            )
        return math.exp(ll)


def search_mappings(
    model,
    tokenizer,
    task_name,
    k,
    seed,
    beam,
    output_dir,
    data_dir,
    content_length,
    mapping_num,
):
    def format_text(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        text = text.replace("</s>", "")
        text = text.replace("▁", " ")
        text = text.strip()
        return text

    print("#", task_name, k, seed, beam)
    dataset_path = os.path.join(data_dir, task_name, "{}-{}".format(k, seed))
    dataset = load_dataset(task_name, dataset_path)
    print("|", "dataset examples")
    print("|", dataset[0])
    print("|", dataset[-1])
    print()

    # Manual label word mappings
    map_of_mapping = {
        "SST-2": {"0": "terrible", "1": "great"},
        "sst-5": {0: "terrible", 1: "bad", 2: "okay", 3: "good", 4: "great"},
        "mr": {0: "terrible", 1: "great"},
        "cr": {0: "terrible", 1: "great"},
        "subj": {0: "subjective", 1: "objective"},
        "trec": {
            0: "Description",
            1: "Entity",
            2: "Expression",
            3: "Human",
            4: "Location",
            5: "Number",
        },
        "mpqa": {0: "terrible", 1: "great"},
        "CoLA": {"0": "incorrect", "1": "correct"},
        "MRPC": {"0": "No", "1": "Yes"},
        "QQP": {"0": "No", "1": "Yes"},
        "STS-B": {"0": "No", "1": "Yes"},
        "MNLI": {"contradiction": "No", "entailment": "Yes", "neutral": "Maybe"},
        "SNLI": {"contradiction": "No", "entailment": "Yes", "neutral": "Maybe"},
        "QNLI": {"not_entailment": "No", "entailment": "Yes"},
        "RTE": {"not_entailment": "No", "entailment": "Yes"},
        "AX-b": {"not_entailment": "No", "entailment": "Yes"},
        "AX-g": {"not_entailment": "No", "entailment": "Yes"},  # TODO: swap
        "BoolQ": {False: "No", True: "Yes"},
        "CB": {"contradiction": "No", "entailment": "Yes", "neutral": "Maybe"},
        "COPA": {0: "choice1", 1: "choice2"},
        "MultiRC": {0: "No", 1: "Yes"},
        "WiC": {False: "No", True: "Yes"},
    }

    template = ""
    mapping = map_of_mapping[task_name]
    print("|", "mapping")
    print("|", mapping)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    mapping_labels = dict()
    for label in mapping.keys():
        f = open(
            os.path.join(output_dir, task_name, "{}-{}-{}.txt".format(k, seed, label)),
            "w",
        )

        if task_name in ["SST-2", "sst-5", "mr", "cr", "subj", "trec", "CoLA", "mpqa"]:
            # Single sentence tasks
            # We take two kinds of templates: put [MASK] at the beginning or the end
            if task_name != "mpqa":
                template = "*cls**sentu_0**<extra_id_0>**sep+*"
            else:
                template = "*cls**sent_0*_Overall_my_impression_is*<extra_id_0>*.*sep+*"
            generate_text = generate(
                dataset,
                template,
                model,
                tokenizer,
                target_number=1,
                mapping=mapping,
                beam=beam,
                content_length=content_length,
                label=label,
            )[:beam]

            print("####### generated label sequences #######")
            for i in range(len(generate_text)):
                # Transform T5 outputs to our template format
                generate_text[i][0] = format_text(generate_text[i][0])
                print(generate_text[i][0])
                f.write(generate_text[i][0] + " " + str(generate_text[i][1]) + "\n")
            print("####### generated label sequences #######\n")
            mapping_labels[label] = generate_text

        elif task_name in [
            "MRPC",
            "QQP",
            "STS-B",
            "MNLI",
            "SNLI",
            "QNLI",
            "RTE",
            "AX-b",
            "AX-g",
            "BoolQ",
            "CB",
            "COPA",
            "MultiRC",
            "WiC",
        ]:
            # Sentence pair tasks
            # We always put [MASK] between the two sentences
            if task_name == "BoolQ":
                template = "*cls**sentu_0*?*<extra_id_0>*,*+sent_1**sep+*"
            elif task_name == "COPA":
                template = (
                    "*cls**sent_0**+sentu_1*?*sent-_2*?*<extra_id_0>*,*+sentl_3**sep+*"
                )
            elif task_name == "MultiRC":
                template = "*cls**sent_1**<extra_id_0>*,*+sentl_2**+sent_0**sep+*"
            elif task_name == "WiC":
                template = "*cls**sent_0**+sent_1*_'*sent_2*'*<extra_id_0>**sep+*"
            else:
                template = "*cls**sent-_0*?*<extra_id_0>*,*+sentl_1**sep+*"
            generate_text = generate(
                dataset,
                template,
                model,
                tokenizer,
                target_number=1,
                mapping=mapping,
                beam=beam,
                content_length=content_length,
                label=label,
            )

            print("####### generated label sequences #######")
            for i in range(len(generate_text)):
                # Transform T5 outputs to our template format
                generate_text[i][0] = format_text(generate_text[i][0])
                print(generate_text[i][0])
                f.write(generate_text[i][0] + " " + str(generate_text[i][1]) + "\n")
            print("####### generated label sequences #######\n")
            mapping_labels[label] = generate_text
        else:
            raise NotImplementedError

    mapping_labels_copy = copy.deepcopy(mapping_labels)
    for key1, value1 in mapping_labels.items():
        for i in range(len(value1)):
            for key2, value2 in mapping_labels_copy.items():
                if key1 != key2:
                    flag = 0
                    for text2, prob in value2:
                        if value1[i][0] == text2:
                            flag = 1
                            value1[i][1] -= prob
                            break
                    if flag == 0:
                        value1[i][1] -= get_prob(
                            dataset,
                            template,
                            model,
                            tokenizer,
                            mapping=mapping,
                            decoder_input=value1[i][0],
                            label=key2,
                        )
        mapping_labels[key1].sort(key=lambda x: x[1], reverse=True)
    f = open(
        os.path.join(output_dir, task_name, "result.json"),
        "w",
    )
    f.write(json.dumps(mapping_labels))

    f = open(
        os.path.join(output_dir, task_name, "16-" + str(seed) + ".txt"),
        "w",
    )
    mapping_list = []

    def gather_mappings(num, now_mapping, prob):
        if num == len(mapping_labels.values()):
            mapping_list.append([copy.deepcopy(now_mapping), prob])
            return
        for i in range(20 if task_name != "trec" else 10):
            now_mapping[list(mapping_labels.keys())[num]] = list(
                mapping_labels.values()
            )[num][i][0]
            gather_mappings(
                num + 1, now_mapping, prob + list(mapping_labels.values())[num][i][1]
            )

    gather_mappings(0, {}, 0)
    mapping_list.sort(key=lambda x: x[1], reverse=True)
    for i in mapping_list[:mapping_num]:
        f.write(str(i[0]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t5_model",
        type=str,
        default="google/t5-v1_1-large",
        help="T5 pre-trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[42, 13, 21, 100, 87],
        help="Data split seeds",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        nargs="+",
        default=[
            "SST-2",
            "sst-5",
            "mr",
            "cr",
            "mpqa",
            "subj",
            "trec",
            "CoLA",
            "MRPC",
            "QQP",
            "STS-B",
            "MNLI",
            "SNLI",
            "QNLI",
            "RTE",
            "AX-b",
            "AX-g",
            "BoolQ",
            "CB",
            "COPA",
            "MultiRC",
            "WiC",
        ],
        help="Task names",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")

    parser.add_argument(
        "--data_dir", type=str, default="data/k-shot", help="Data directory"
    )
    parser.add_argument("--beam", type=int, default=50, help="Beam search width")
    parser.add_argument(
        "--k", type=int, default=16, help="Number of training instances per label"
    )
    parser.add_argument(
        "--content_length",
        type=int,
        default=20,
        help="Content length",
    )
    parser.add_argument(
        "--mapping_num", type=int, default=20, help="Number of mappings"
    )

    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.sep_token = "</s>"

    model = model.cuda()
    model.eval()

    for task_name in args.task_name:
        for seed in args.seed:
            search_mappings(
                model=model,
                tokenizer=tokenizer,
                task_name=task_name,
                k=args.k,
                seed=seed,
                beam=args.beam,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                content_length=args.content_length,
                mapping_num=args.mapping_num,
            )


if __name__ == "__main__":
    main()
