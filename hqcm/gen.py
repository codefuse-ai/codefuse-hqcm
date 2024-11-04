import json
import pprint
import time

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()


class SFTModel:

    def __init__(
        self, model_id, peft_adapter_id=None,
        max_response_length=512, max_prompt_length=4096,
        temperature=0.8, top_k=50, top_p=0.95
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto",
            trust_remote_code=True
        )
        if peft_adapter_id is not None:
            self.model.load_adapter(peft_adapter_id)
        self.generation_config = {
            "max_new_tokens": max_response_length,
            "num_return_sequences": 1,
            "num_beams": 1
        }
        if abs(temperature) < 1e-8:  # temperature is zero
            self.generation_config["do_sample"] = False
        else:
            self.generation_config["do_sample"] = True
            self.generation_config["temperature"] = temperature
            self.generation_config["top_k"] = top_k
            self.generation_config["top_p"] = top_p

    def query(self, prompt):
        inp_tk = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # TODO: What if len(inp_tk) > self.max_prompt_length?
        oup_tk = self.model.generate(**inp_tk, **self.generation_config).to("cpu")
        answer = self.tokenizer.batch_decode(
            oup_tk[:, inp_tk['input_ids'].shape[-1]:],  # Don't echo the preceding chat_message in the answer
            skip_special_tokens=True,  # Skip special tokens like EOS, PAD in answer
            clean_up_tokenization_spaces=False)[0]
        return answer


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(
        prog="gen",
        description="Call a supervised fine-tuned model to generate answers for the transformed HQCM dataset"
    )
    parser.add_argument(
        'adapter',
        type=Path,
        help="Path to the adapter after supervised fine-tuning, or to the plain model if -M is specified."
    )
    parser.add_argument(
        "-d", "--dataset",
        required=True, type=str,
        help="Path to the dataset for inference"
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
        '-o', '--output',
        required=True, type=Path,
        help="Path to the JSON file saving the inference results"
    )
    parser.add_argument(
        '-T', '--temperature',
        default=0.8, type=float,
        help="Temperature set to the model, controlling the randomness of the model"
    )
    parser.add_argument(
        '-M', '--plain-model',
        default=False, action='store_true',
        help="Loading the plain model without considering the adapters (i.e., the adapter points to a model, not an adapter)"
    )

    args = parser.parse_args()
    pprint.pprint(vars(args))

    # TODO: Directly load the model with the adapter id
    if not args.plain_model:
        adapter_config_path = args.adapter / 'adapter_config.json'
        assert adapter_config_path.exists(), f'Adapter config does not exists under {adapter_config_path}'
        with adapter_config_path.open('r') as fin:
            adapter_config = json.load(fin)
        model_path = adapter_config['base_model_name_or_path']

        model = SFTModel(
            model_id=model_path,
            peft_adapter_id=str(args.adapter.absolute()),
            max_prompt_length=512,
            temperature=args.temperature
        )
    else:
        model_path = str(args.adapter)
        model = SFTModel(
            model_id=model_path,
            max_prompt_length=512,
            temperature=args.temperature
        )

    results = []
    for index, item in enumerate(load_dataset(args.dataset, args.config, split=args.split)):
        prompt = item['prompt']
        answer = item['answer']

        start_time_ms = time.time() * 1000
        try:
            model_answer = model.query(prompt)
        except Exception as e:
            model_answer = f'Generation Failed: {e}'
        end_time_ms = time.time() * 1000

        elapsed_ms = (end_time_ms - start_time_ms)

        results.append({
            'prompt': prompt,
            'expected_answer': answer,
            'actual_answer': model_answer,
            'elapsed_ms': elapsed_ms
        })

        print(f"#{index} (elapsed: {elapsed_ms}ms) {answer}   :   {model_answer}")

    with args.output.open("w", encoding='utf-8') as fou:
        json.dump(results, fou, ensure_ascii=False, indent=2)
