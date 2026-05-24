#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-./data}"
MODEL_ROOT="${MODEL_ROOT:-./models/unlearn}"

METHOD_NAMES=(${METHOD_NAMES:-GradAscent GradDiff NPO SimNPO RMU WGA UNDIAL SatImp})
# METHOD_NAMES=(${METHOD_NAMES:-RAZOR})
SFT_SAMPLE_SIZES=(${SFT_SAMPLE_SIZES:-200 400 600 800 1000})
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
for METHOD_NAME in "${METHOD_NAMES[@]}"; do
	python src/test/vgg_test_retrain_ood.py \
		--input-parquet "${INPUT_PARQUET:-${DATA_ROOT}/facerec/facerec_test.parquet}" \
		--sft-parquet "${SFT_PARQUET:-${DATA_ROOT}/pacs/qwen2.5_vl_3b_train_1over20.parquet}" \
		--sft-sample-sizes "${SFT_SAMPLE_SIZES[@]}" \
		--sample-seed "${SAMPLE_SEED:-0}" \
		--model-path "${MODEL_PATH:-${MODEL_ROOT}/rec_Qwen2.5-VL-3B-Instruct_${METHOD_NAME}}" \
		--model-name "${MODEL_NAME:-qwen2_5-vl-3b}" \
		--method-name "${METHOD_NAME}" \
		--output-root "${OUTPUT_ROOT:-./results_ood_attack}" \
		--tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
		--sft-nproc-per-node "${SFT_NPROC_PER_NODE:-4}" \
		"$@"
done
