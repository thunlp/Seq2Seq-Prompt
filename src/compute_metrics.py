# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
from collections import defaultdict, Counter

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    import numpy as np
    from scipy.special import softmax

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels, f1_avg="binary"):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def superglue_compute_metrics(task_name, preds, labels, guids=None, answers=None):
        assert len(preds) == len(labels)
        if task_name == "boolq":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "cb":
            return acc_and_f1(preds, labels, f1_avg="macro")
        elif task_name == "copa":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "multirc":
            assert len(guids) == len(preds), "Different number of predictions and IDs!"
            qst2ans = defaultdict(list)
            # iterate over examples and aggregate statistics
            for idx, pred, label in zip(guids, preds, labels):
                qst_idx = f"{idx[0]}-{idx[1]}"
                qst2ans[qst_idx].append((pred, label))

            f1s, ems = [], []
            for qst, preds_and_labels in qst2ans.items():
                preds, labels = zip(*preds_and_labels)
                f1 = f1_score(y_true=labels, y_pred=preds)
                f1s.append(f1)
                em = int(
                    sum([p == l for p, l in preds_and_labels]) == len(preds_and_labels)
                )
                ems.append(em)

            return {"f1": sum(f1s) / len(f1s), "em": sum(ems) / len(ems)}
        elif task_name == "record":
            f1s, ems = [], []
            for f1, em in preds:
                f1s.append(f1)
                ems.append(em)

            return {"f1": sum(f1s) / len(f1s), "em": sum(ems) / len(ems)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wic":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wsc":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
