# CodeFuse HQCMBench

HQCM is a small-scale yet high-quality dataset designed for *Code-Change Understanding*.
It is a carefully developed subset of the Java portion of the [MCMD](https://doi.org/10.1007/s10664-022-10219-1) dataset.
Each entry in HQCM has been meticulously selected, reviewed, revised, and validated by crowdsource developers.
The creation of HQCM stems from the recognition that large language models (LLMs) are not silver bullets;
there are scenarios where their application may be limited, for example:

1. **Security Constraints**: In cases where data security is paramount, commercial LLMs are often prohibited to prevent potential data leaks, especially in industrial settings.
2. **Compute Constraints**: LLMs are often difficult to deploy in resource-constrained environments, such as laptops and mobile devices at the edge.
3. **Financial Constraints**: The high cost of premium LLM APIs makes their use infeasible for many applications without enough budgets.
4. **Customized Tasks**: LLMs' performance, especially those non-premium ones, can vary significantly across specialized or customized tasks.

In these contexts, HQCM aims to serve as training and testing data for SLMs (small language models), or as few-shot examples for LLMs in tasks involving code-change understanding.

HQCM comprises approximately 5,000 high-quality pairs of code changes and their corresponding summaries, where each code change is presented in a unified diff format, while the accompanying summary is a concise sentence available in both English and Chinese.
Each entry in HQCM is classified into one of eight popular categories: *feat* (feature), *fix*, *refactor*, *cicd* (CI/CD), *build*, *test*, *docs* (documentation), and *style*.
Additional categories such as *perf* (performance) and *chore* are planned for future inclusion.
The distribution of these categories reflects their natural prevalence in the real world, with refactor being the most common and style and CI/CD being the least prevalent.

## Installation

```shell
git clone https://github.com/codefuse-ai/codefuse-hqcm hqcm && cd hqcm
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Task Adaptation 

HQCM can be adapted for three change-related tasks:
- **Change Summarization** (`chsum`) summarizes a code change (represented by a code diff) into a short sentence in natural language
- **Change Classification** (`chcl`) classifies each pair of code change and summary into one of the categories
- **Code Refinement** (`coderef`) refines a given piece of code based a comment to produce the refined code, commonly used in code review process 

Below transforms HQCM into its `chsum` (change-summarization) variant and saves the variant into `$CHSUM_PATH` for supervised fine-tuning:

```shell
export CHSUM_VARIANT_PATH='./dataset/chsum'
python -m hqcm.xdata --task chsum --output $CHSUM_VARIANT_PATH ./dataset/
```

## Fine-tuning SLMs

The adapted dataset can be used for SLMs' supervised fine-tuning or used as few-shot examples for LLMs.
We provided scripts to fine-tune a HuggingFace model with LoRA based on the transformed dataset.

Below fine-tunes Llama2-7b for change summarization and saves it into `$CHSUM_MODEL_PATH`, using HQCM's chsum variant in `$CHSUM_VARIANT_PATH`:

```sh
export CHSUM_MODEL_PATH='/path/to/chsum_model'
python -m hqcm.sft                \
    --seed 0                      \
    --learning-rate '2e-4'        \
    --num-epochs 5                \
    --batch-size 1                \
    --micro-batch-size 1          \
    --lora-rank 64                \
    --lora-alpha 16               \
    --lora-dropout 0.1            \
    --quantization '-1'           \
    --dataset $CHSUM_VARIANT_PATH \
    --split 'train'               \
    --max-length 512              \
    --output $CHSUM_MODEL_PATH    \
    '/path/to/your/llama2-7b'
```

Below leverages the fine-tuned model in `$CHSUM_MODEL_PATH` to generate summaries for changes in the test split of `$CHSUM_VARIANT_PATH`, with results exporting to `$CHSUM_RES_PATH`:

```shell
export CHSUM_RES_PATH='/path/to/chsum_results'
python -m hqcm.gen                \
    --dataset $CHSUM_VARIANT_PATH \
    --split 'test'                \
    --output $CHSUM_RES_PATH      \
    --temperature 0               \
    $CHSUM_MODEL_PATH
```

The above scripts also support HQCM's chcl and coderef variants.

## FAQs

Q: The fine-tuning and generation scripts got stuck when connecting to HuggingFace
A: HQCM's scripts assume an offline environment. Perhaps disabling download by:

```shell
export HF_DATASETS_OFFLINE=1         # Disable HuggingFace's online accessing to datasets
export TRANSFORMERS_OFFLINE=1        # Disable HuggingFace's online accessing to models
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer's parallelism
```

Q: Does HQCM support other code change-related tasks?
A: HQCM is code-change dataset. Users can adapt it to any change-related tasks in theory, but we did not experiment this. This require users to comprehend the task and reformat the dataset according to their usages. We are expecting promising results and we welcome such adaptations.

## Citation

HQCM was published in [ASE '24](https://dl.acm.org/doi/10.1145/3691620.3694999).
If you find it helpful, please consider citing our paper:

```txt
@inproceedings{hqcm_ase24,
  author = {Li, Cong and Xu, Zhaogui and Di, Peng and Wang, Dongxia and Li, Zheng and Zheng, Qian},
  title = {Understanding Code Changes Practically with Small-Scale Language Models},
  year = {2024},
  isbn = {9798400712487},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3691620.3694999},
  doi = {10.1145/3691620.3694999},
  booktitle = {Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering},
  pages = {216–228},
  numpages = {13},
  keywords = {code change, code review, language model, LLM, SLM},
  location = {Sacramento, CA, USA},
  series = {ASE '24}
}
```
