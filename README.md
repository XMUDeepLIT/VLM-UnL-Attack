# Unlearning Attack

This folder contains the VGG retrain attack workflow.

## Installation

Install dependencies before running the examples:

```bash
pip install -r requirements.txt
```

Download LlamaFactory from the released repository, then place it under `src/LlamaFactory`:

```bash
git clone <ANONYMOUS_LLAMAFACTORY_URL> src/LlamaFactory
```

If you receive LlamaFactory as an archive instead, extract it so that the package root is `src/LlamaFactory`.

## One-Command Launch

Run both IND and OOD retrain attacks:

```bash
bash run_vgg_retrain.sh
```

Run one side only:

```bash
bash run_vgg_retrain.sh ind
bash run_vgg_retrain.sh ood
```

Extra arguments are passed to the Python entrypoints:

```bash
bash run_vgg_retrain.sh ind --overwrite
METHOD_NAMES="UNDIAL SatImp" bash run_vgg_retrain.sh ood
```

## Included Files

- `scripts/vgg/vgg_test_retrain_ind.sh`
- `scripts/vgg/vgg_test_retrain_ood.sh`
- `src/test/vgg_test_retrain_ind.py`
- `src/test/vgg_test_retrain_ood.py`
- `requirements.txt`

## External Inputs

By default, the launch scripts look for datasets and models under project-local folders:

- `DATA_ROOT=./data`
- `MODEL_ROOT=./models/unlearn`
- `INPUT_PARQUET=${DATA_ROOT}/facerec/facerec_test.parquet`
- `FORGET_PARQUET=${DATA_ROOT}/facerec/qwen2.5_vl_3b_train_forget.parquet`
- `SFT_PARQUET=${DATA_ROOT}/pacs/qwen2.5_vl_3b_train_1over20.parquet`
- `MODEL_PATH=${MODEL_ROOT}/rec_Qwen2.5-VL-3B-Instruct_${METHOD_NAME}`

Place each already-trained unlearned model under the matching `MODEL_PATH` directory, for example `./models/unlearn/rec_Qwen2.5-VL-3B-Instruct_mmunlearner`.

The PACS SFT parquet above is a 1/20-size subset used as a lightweight example.

You can override them with environment variables before running the script.
