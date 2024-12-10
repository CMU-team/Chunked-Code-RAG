# Chunked CodeRAG-Bench

This is the code repository for the project ["Chunked CodeRAG-Bench - Improving Retrieval Augmented Generation in Code using Chunking"] based on the CodeRAG-Bench (https://code-rag-bench.github.io/) for open domain programming tasks for ANLP Assignment 4(11-711) Fall 2024.

## Installation

Create a new environment:
```
conda env create -n crag python=3.10 -y
conda activate crag
```
And install the necessary libraries:
```
pip install -r requirements.txt
```

## Organization
- [Retrieval](retrieval/): Code to run retrieval, with BM25, dense retrievers via [sentence-transformers](https://www.sbert.net/), and proprietary API embeddings.
- [Generation](generation/): Code to run model generation and execution-based evaluation.
- [Preprocess](preprocessor/): code to preprocess raw data for retrieval pool construction, see inside the directory for details.

## Pre-Retrieval Chunking

```
cd retrieval/
```

### Fixed Size Chunking

```
python fixed-size-chunking.py
    --result_path datasets/PATH_TO_YOUR_DATASET \
    --tokenizer TOKENIZER \
    --max_num_tokens CHUNK SIZE
```

### Semantic Chunking

```
python semantic-chunking.py
    --result_path datasets/PATH_TO_YOUR_DATASET \
    --tokenizer TOKENIZER
```

## Retrieval

### Dataset Preprocessing
Before running retrieval on a dataset, you need to create the datastore for it. Following
```
python -m create/${data_name}.py
# choices for ${data_name}
# open-domain: 'ds1000', 'odex'
```

### Run dense embedding models

#### Retrieval using canonical retrieval source
Run your embedding models by loading embedding models from `sentence-transformers` as follows:

```sh
python3 eval_beir_sbert_canonical.py \
    --model YOUR_MODEL_NAME_OR_PATH \
    --dataset TASK_NAME \
    --output_file PATH_TO_YOUR_SCORE_FILE \
    --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
```
By specifying the output file name `--output_file`, you can save the retrieval results as a json file.

```json
{'ndcg': {'NDCG@1': 0.61667, 'NDCG@3': 0.68203, 'NDCG@5': 0.70804, 'NDCG@10': 0.72701, 'NDCG@100': 0.74926, 'NDCG@1000': 0.75551}, 'mrr': {'MRR@1': 0.61667, 'MRR@3': 0.67278, 'MRR@5': 0.68611, 'MRR@10': 0.69368, 'MRR@100': 0.69721, 'MRR@1000': 0.69744}, 'recall': {'Recall@1': 0.58817, 'Recall@3': 0.728, 'Recall@5': 0.79294, 'Recall@10': 0.84789, 'Recall@100': 0.95, 'Recall@1000': 0.99667}, 'precision': {'P@1': 0.61667, 'P@3': 0.26444, 'P@5': 0.176, 'P@10': 0.09533, 'P@100': 0.01077, 'P@1000': 0.00113}}
```

`--results_file` indicates the file name to store retrieval results, which will be used in the subsequent RAG evaluations.

### Run BM25 
Start with a fresh environment with python 3.10
```sh
# install pyserini
pip install pyserini==0.25.0
# install openjdk-11 and maven (if you don't have any)
conda install -c conda-forge openjdk=11 maven -y
```
For more information of installing pyserini, please refer to [installation guide for pyserini](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

#### For canonical retrieval source (non-repo)

Preprocess all the corpus file of existing datasets into pyserini indexable format. For each dataset, the modified corpus will be saved in `OUTPUT_DIR/{DATASET_NAME}_corpus/edit.jsonl`:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --stage preprocess
```

Indexing the corpus from `OUTPUT_DIR/{DATASET_NAME}_corpus/edit.jsonl`, and the index will be saved in `INDEX_DIR/{DATASET_NAME}_corpus/`:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --stage index
```

Search the query from the target dataset using BM25:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --stage search
```
The score file will be saved in `results/{DATASET_NAME}_k1={K1}_b={B}_pyserini_bm25_output.jsonl`, retrieval results in `results/{DATASET_NAME}_k1={K1}_b={B}_pyserini_bm25.jsonl`

## Generation

The `main.py` script supports running code generation with any models supported by huggingface or OpenAI.

### Baseline Generation
To run no-retrieval generation on the orignal dataset, specify its huggingface dataset name in the `dataset_path` argument:
```bash
python main.py --task "humaneval" \
--model "bigcode/starcoder2-7b" \
--dataset_path "openai_humaneval" \
--allow_code_execution
```
Set `--allow_code_execution` to evaluate generations with code execution, this is required for all tasks.

Note that the `task` should align with the `dataset_path`. All tasks available are:
- open domain: 'ds1000-all-completion', 'odex-en'

### Retrieval-Augmented Code Generation
Running generation with previous retrieval results, e.g., "retrieval/humaneval/gist_large.json", specify the files as follows:
```bash
python main.py --task "humaneval" \
--model "bigcode/starcoder2-7b" \
--dataset_path "json" --data_files_test "retrieval/humaneval/gist_large.json" \
--allow_code_execution
```

Running the `main.py` script will automatically conduct execution-based evaluation after the generation is finished.

### References

Wang, Z. Z., Asai, A., Yu, X. V., Xu, F. F., Xie, Y., Neubig, G., & Fried, D. (2024). CodeRAG-Bench: Can Retrieval Augment Code Generation?. arXiv preprint arXiv:2406.14497.
