import json
import sys

PROMPT_TEMPLATE="""\
Please generate a commit message for the following diff:

```diff
{diff}
```
"""


def transform_item(item):
    return {
        'prompt': PROMPT_TEMPLATE.format(diff=item['change']),
        'answer': item['summaries']['en']
    }


def transform(in_dir, out_dir):
    assert in_dir.is_dir(), f"Not a directory: {in_dir}"
    assert (in_dir / 'train.json').exists(), f"File train.json does not exist in: {in_dir}"
    assert (in_dir / 'test.json').exists(), f"File test.json does not exist in: {in_dir}"

    out_dir.mkdir(exist_ok=True)

    for fname in ['train.json', 'test.json']:
        with (in_dir / fname).open('r') as fin:
            tx_data = [transform_item(item) for item in json.load(fin)]
        with (out_dir / fname).open('w') as fou:
            json.dump(tx_data, fou, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(
        prog="tx_chsum",
        description="Tranform the HQCM dataset for finetuning of change summarization"
    )
    parser.add_argument(
        "dataset", type=Path,
        help="Path to the directory saving the HQCM dataset before transformation"
    )
    parser.add_argument(
        "-o", "--output",
        required=True, type=Path,
        help="Path to the directory to save the HQCM dataset after transforming for change summarization"
    )
    args = parser.parse_args()

    transform(args.dataset, args.output)
