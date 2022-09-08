# Seq2Seq-Prompt

Source code and dataset for "Automatic Label Sequence Generation for Prompting Sequence-to-sequence Models".

## Quick links

- [Overview](#overview)
- [Requirements](#requirements)
- [Prepare the data](#prepare-the-data)
- [Run AutoSeq](#run-autoseq)
	-	[Main results](#main-results)
	-	[Search for automatic label sequence and label word mappings](#search-for-automatic-label-sequence-and-label-word-mappings)
- [Citation](#citation)
## Overview

In this work we propose AutoSeq, a general-purpose, fully automatic prompting method that depends only on few-shot annotations.

![](./figs/AutoSeq.png)

Figure: Illustration of AutoSeq. We first use T5 to generate label sequence candidates given each label's training instances; we then use contrastive re-ranking to get label sequences that are more label-specific; in the end we enumerate all the combinations and re-rank by the fine-tuning performance.

## Requirements

To run our code, please install all the dependencies:

```
pip install -r requirements.txt
```

## Prepare the data

Please download the official GLUE and SuperGLUE data and extract the files to `./data/GLUE` and `./data/SuperGLUE`.

Then use the following command to generate the few-shot data:

```bash
python tools/generate_k_shot_data_for_GLUE.py
python tools/generate_k_shot_data_for_SuperGLUE.py
```

## Run AutoSeq

For convenience, we use SST-2 as an example below.

### Main results

AutoSeq:

```bash
task=SST-2 bash scripts/t5_seq.sh
```

AutoWord:

```bash
task=SST-2 bash scripts/t5_word.sh
```

Manual:

```bash
task=SST-2 bash scripts/t5_man.sh
```

Manual with eng.:

```bash
task=SST-2 bash scripts/t5_man_eng.sh
```

Fine-tuning:

```bash
task=SST-2 bash scripts/t5_fine_tuning.sh
```

Fine-tuning (Full train set):

```bash
task=SST-2 bash scripts/t5_full_fine_tuning.sh
```

Prompt tuning:

```bash
task=SST-2 bash scripts/t5_prompt_tuning.sh
```

All the results will be stored in `./log`. To gather all the results, run the following command:

```bash
python tools/gather_result.py --condition "{'tag': 'exp-SPL-SST-2-auto', 'task_name': 'sst-2', 'few_shot_type': 'prompt'}"
```

### Search for automatic label sequence and label word mappings

You can find our generated mappings in `./auto_label_sequences` and `./auto_label_words`.

We first generate candidate label sequence and label word mappings by running:

```bash
python tools/generate_label_sequences.py
python tools/generate_label_words.py
```

Then we do prompt-based fine-tuning of all the mappings by:

```bash
task=SST-2 bash scripts/sort_mapping.sh
```

Finally we sort them based on dev set performance:

```bash
python tools/sort_mapping.py --condition "{'tag': 'exp-mapping-SST-2', 'task_name': 'sst-2'}" --mapping_dir auto_label_sequences
```

## Citation

Please cite our paper if you use AutoSeq in your work:

```bibtex
@inproceedings{yu2022automatic,
   title={Automatic Label Sequence Generation for Prompting Sequence-to-sequence Models},
   author={Zichun Yu and Tianyu Gao and Zhengyan Zhang and Yankai Lin and Zhiyuan Liu and Maosong Sun and Jie Zhou},
   booktitle={International Conference on Computational Linguistics (COLING)},
   year={2022}
}
```