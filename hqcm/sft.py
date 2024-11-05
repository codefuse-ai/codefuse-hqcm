import os
import pprint
import time
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, set_seed, AutoModelForCausalLM, TrainingArguments, \
    Trainer, BitsAndBytesConfig

load_dotenv()

parser = ArgumentParser(
    "sft",
    description="Supervised fine-tuning (sft) a HuggingFace model with bf16 precision for "
                "a specific code-change related task using the transformed HQCM dataset"
)
parser.add_argument(
    "model",
    type=str,
    help="Path to the base model for sft"
)
parser.add_argument(
    "-d", "--dataset",
    required=True, type=str,
    help="Path to the dataset for base model's sft"
)
parser.add_argument(
    "-c", "--config",
    default=None, type=str,
    help="Config name of the dataset"
)
parser.add_argument(
    "-t", "--split",
    default=None, type=str,
    help="Split of the dataset; each item in the dataset should have a `prompt` and an `answer` field"
)
parser.add_argument(
    "-p", "--percentage",
    default=100, type=int,
    help="Select only a percentage of the dataset for sft (e.g., 10 for 10%)"
)
parser.add_argument(
    "-M", "--max-length",
    default=512, type=int,
    help="Max length of the text (prompt + answer) saved in the dataset"
)
parser.add_argument(
    "-N", "--no-lora",
    default=False, action="store_true",
    help="Fine-tuning without LoRA"
)
parser.add_argument(
    "-R", "--lora-rank",
    default=64, type=int,
    help="LoRA's rank"
)
parser.add_argument(
    "-A", "--lora-alpha",
    default=16, type=int,
    help="LoRA's alpha"
)
parser.add_argument(
    "-D", "--lora-dropout",
    default=0.1, type=float,
    help="LoRA's dropout"
)
parser.add_argument(
    "-Q", "--quantization",
    default=-1, type=int, choices=[-1, 4, 8],
    help="Enable fine-tuning with k-bit quantization"
)
parser.add_argument(
    "-a", "--learning-rate",
    default=2e-4, type=float,
    help="Learning rate (the larger, the fast)"
)
parser.add_argument(
    "-e", "--num-epochs",
    default=2, type=int,
    help="Number of epochs to train"
)
parser.add_argument(
    "-B", "--batch-size",
    default=64, type=int,
    help="Number of examples to feed for gradient updates (i.e., opt_per_device_batch_size * gradient_accumulation_steps)"
)
parser.add_argument(
    "-b", "--micro-batch-size",
    default=8, type=int,
    help="Number of examples to feed per step (i.e., per_device_batch_size)"
)
parser.add_argument(
    "-r", "--resume",
    default=False, action='store_true',
    help="Resume pretraining from the last checkpoint in the output directory"
)
parser.add_argument(
    "-o", "--output",
    required=True, type=str,
    help="Path to saved the model after sft"
)
parser.add_argument(
    "-s", "--seed",
    default=int(time.time()), type=int,
    help="Seed for controlling the randomness of the sft process"
)

args = parser.parse_args()
pprint.pprint(vars(args))

arg_base_model = args.model

arg_dataset = args.dataset
opt_dataset_config = args.config
opt_dataset_split = args.split
opt_dataset_max_seq_len = args.max_length
opt_dataset_percentage = args.percentage

opt_lora = not args.no_lora
opt_lora_rank = args.lora_rank
opt_lora_alpha = args.lora_alpha
opt_lora_dropout = args.lora_dropout

opt_quant_bit = args.quantization

opt_learning_rate = args.learning_rate
opt_training_epochs = args.num_epochs
opt_per_device_batch_size = args.micro_batch_size
opt_gradient_accumulation_steps = args.batch_size // args.micro_batch_size

opt_resume_from_checkpoint = args.resume

arg_output_dir = args.output
opt_output_save_steps = 1000
opt_output_save_limit = 5
opt_output_logging_steps = 10

opt_seed = args.seed
opt_num_proc = os.cpu_count()

# Set the seed for reproduction
set_seed(opt_seed)

# Write out the command for reproduction
os.makedirs(arg_output_dir, exist_ok=True)
with open(os.path.join(arg_output_dir, "command.txt"), 'w') as fp:
    fp.write(pprint.pformat(vars(args)))

# Load the tokenizer and set the pad token
tokenizer = AutoTokenizer.from_pretrained(arg_base_model, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = "left"

# Create a DataCollator to help us pad each sequence to the maximum length in the batch while training.
# Don't use DataCollatorForLanguageModeling as it doesn't pad the labels.
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt"
)


# Tokenize the dataset and hide the prompt from the model
def tokenize_and_preprocess(example):
    prompt, answer = example['prompt'], example['answer']

    tk_example = tokenizer(
        prompt + answer,
        max_length=opt_dataset_max_seq_len,
        # Don't pad since we are about to pad each example by the DataCollator
        padding=False, truncation=True
    )

    # In case there lacks an EOS token. This is because the tokenizer is
    # used for generation tasks, so it automatically adds a <bos> token to the
    # head but does not add a EOS token to the end for further generation.
    # So let's add it as the end of our prompt-answer pair.
    if (tk_example['input_ids'][-1] != tokenizer.eos_token_id and
        len(tk_example['input_ids']) < opt_dataset_max_seq_len):
        tk_example['input_ids'].append(tokenizer.eos_token_id)
        tk_example['attention_mask'].append(1)  # This EOS token should be attended

    # Prepare our labels, it's exactly the input_ids
    tk_example['labels'] = tk_example['input_ids'].copy()

    # Hide the prompt such that our training process does not compute cross-entropy for prompts
    # and our model only focuses on learning to generate the answer.
    # Since the last token of the prompt might be part of the first token of the answer, let's skip it.
    # For example, if prompt="A " and answer="a" then tk_prompt=["_A", "_"], tk_example=["_A", "_a"].
    num_hidden_tokens = len(tokenizer(
        prompt, max_length=opt_dataset_max_seq_len, padding=False, truncation=True
    )['input_ids']) - 1
    # label_pad_token_id (-100) is a magic number used by pytorch to hide tokens for cross-entropy.
    tk_example['labels'][:num_hidden_tokens] = [data_collator.label_pad_token_id] * num_hidden_tokens

    return tk_example


# Load the dataset for supervised fine-tuning
raw_dataset = load_dataset(arg_dataset, opt_dataset_config, split=opt_dataset_split)
if 0 < opt_dataset_percentage < 100:
    num_selected = int(len(raw_dataset) * (opt_dataset_percentage / 100))
    raw_dataset = raw_dataset.shuffle().select(range(num_selected))
dataset = raw_dataset.map(
    tokenize_and_preprocess,
    remove_columns=raw_dataset.column_names,
    num_proc=opt_num_proc,
    load_from_cache_file=True
)

# Enable quantization if needed
if opt_quant_bit == -1:
    quantization_config = None
elif opt_quant_bit == 4:
    quantization_config = BitsAndBytesConfig(
        # According to https://huggingface.co/blog/4bit-transformers-bitsandbytes:
        # A rule of thumb is: use double quant if you have problems with memory,
        # use NF4 for higher precision, and use a 16-bit dtype for faster finetuning.
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
elif opt_quant_bit == 8:
    quantization_config = BitsAndBytesConfig(
        # TODO: Enable more 8bit options
        load_in_8bit=True
    )
else:
    assert False, f"Unsupported quantization bits: {opt_quant_bit}"

# Load the base model for supervised fine-tuning
base_model = AutoModelForCausalLM.from_pretrained(
    arg_base_model,
    device_map='auto',  # Let the accelerator module decides
    # Let the base model decides, i.e., uses 'torch.dtype' in config.json.
    # If this is not set and using the default value, the model will be loaded by float32.
    torch_dtype='auto',
    # Let's enable quantization during supervised fine-tuning
    quantization_config=quantization_config,
    trust_remote_code=True
)

# Create a LoRA adapter with the PEFT module
if opt_lora:
    lora_adapter = LoraConfig(
        r=opt_lora_rank,
        lora_alpha=opt_lora_alpha,
        lora_dropout=opt_lora_dropout,
        # APIs like base_model.modules() or base_model.named_modules() can help.
        # Each module can be a full module name, a suffix of the name, or a regex.
        # By default, PEFT/LoRA treat each in the following as a suffix of a module
        # name and check against all modules in the model by creating a regex r'.*\.{target_module}$'.
        # If not specified, some default values are used according to different base models.
        # Like for ChatGLM-series models, defaults to ['query_key_value'];
        # Like for Llama-series models, defaults to ['q_proj', 'k_proj', 'v_proj'];
        # Check: TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING in site-packages/peft/utils/others.py.
        # target_modules=['q_proj', 'k_proj', 'v_proj'],  # TODO: support customizing LoRA's target modules
        inference_mode=False,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
else:
    lora_adapter = None
base_model = get_peft_model(base_model, lora_adapter)

# Prepare the arguments for supervised fine-tuning
training_args = TrainingArguments(
    output_dir=arg_output_dir,
    overwrite_output_dir=False,

    # logging arguments
    report_to="tensorboard",
    logging_steps=opt_output_logging_steps,
    logging_first_step=True,
    logging_dir=arg_output_dir,

    # saving arguments
    save_strategy="steps",
    save_steps=opt_output_save_steps,
    save_total_limit=opt_output_save_limit,

    # learning arguments
    learning_rate=opt_learning_rate,
    num_train_epochs=opt_training_epochs,
    per_device_train_batch_size=opt_per_device_batch_size,
    gradient_accumulation_steps=opt_gradient_accumulation_steps,
    # optim=None,
    weight_decay=0.01,  # This adds an additional, regulation item to AdamW to avoid overfitting
    warmup_ratio=0.01,  # This is for learning rate scheduler
    # load_best_model_at_end=True,

    # model arguments
    bf16=True,  # Enforce bf16 precision

    # data loading arguments
    dataloader_drop_last=False,
    dataloader_num_workers=opt_num_proc,
)

# Define the trainer for supervised fine-tuning
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=dataset,
    # TODO support on-the-fly evaluation
    # eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
    data_collator=data_collator
)

# TODO: Check out this line
base_model.config.use_cache = False

# Start supervised fine-tuning
trainer.train(resume_from_checkpoint=True if opt_resume_from_checkpoint else None)

# Save the model
trainer.save_model(arg_output_dir)

print(f"----------")
print(f"Supervised fine-tuning finished; the model was saved to {arg_output_dir}")
