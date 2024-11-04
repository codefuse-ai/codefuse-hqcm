import json
import sys
from unidiff import PatchSet


CODE_OPT_PROMPT_TEMPLATE = """\
// Please optimize the given "Code (to optimize)" (a portion of some "File"s) by strictly following the given "Suggestion".
// Your optimization can involve editing or removing existing code, or adding new code.
// You may pay more attention to lines marked by "// !!attention". If no such marked lines, do everything by yourself.

// Code (to optimize):

{code_to_opt}

// Suggestion: {suggestion}

// Code (after optimization):

"""

FEWSHOT_CODE_OPT_PROMPT_TEMPLATE = """\
// Please optimize the given "Code (to optimize)" (a portion of some "File"s) by strictly following the given "Suggestion".
// Your optimization can involve editing or removing existing code, or adding new code.
// You may pay more attention to lines marked by "// !!attention". If no such marked lines, do everything by yourself.

-------

// Code (to optimize):

//// File: util/src/com/intellij/util/containers/SLRUMap.java
69 |   public void put(K key, V value) {{ 
70 |     V oldValue = myProtectedQueue.remove(key); 
71 |     if (oldValue != null) {{ 
72 |       onDropFromCache(key, value); // !!attention
73 |     }} 
74 |  
75 |     oldValue = myProbationalQueue.put(getStableKey(key), value); 
76 |     if (oldValue != null) {{ 
77 |       onDropFromCache(key, value); // !!attention
78 |     }} 
79 |   }} 
80 |  

// Suggestion: Corrected parameter error in onDropFromCache() function call

// Code (after optimization):

//// File: util/src/com/intellij/util/containers/SLRUMap.java
69 |   public void put(K key, V value) {{
70 |     V oldValue = myProtectedQueue.remove(key);
71 |     if (oldValue != null) {{
72 |       onDropFromCache(key, oldValue);
73 |     }}
74 | 
75 |     oldValue = myProbationalQueue.put(getStableKey(key), value);
76 |     if (oldValue != null) {{
77 |       onDropFromCache(key, oldValue);
78 |     }}
79 |   }}
80 | 

-------

// Code (to optimize):

{code_to_opt}

// Suggestion: {suggestion}

// Code (after optimization):

"""


def transform_item(item, fewshot):
    opt_suggestion = item['summaries']['en']

    code_to_opt = []
    code_after_opt = []

    for patched_file in PatchSet(item['change']):
        file_name = patched_file.path
        source_text = "\n".join(
            f"{line.source_line_no} | {line.value.rstrip()} {'// !!attention' if line.is_removed else ''}"
            for hunk in patched_file 
            for line in hunk.source_lines() 
        )
        target_text = "\n".join(
            f'{line.target_line_no} | {line.value.rstrip()}'
            for hunk in patched_file
            for line in hunk.target_lines()
        )
        code_to_opt.append(f"""//// File: {file_name}\n{source_text}""")
        code_after_opt.append(f"""//// File: {file_name}\n{target_text}""")

    if fewshot:
        return {
            'prompt': FEWSHOT_CODE_OPT_PROMPT_TEMPLATE.format(
                code_to_opt='\n'.join(code_to_opt),
                suggestion=opt_suggestion
            ),
            'answer': '\n'.join(code_after_opt)
        }
    else:
        return {
            'prompt': CODE_OPT_PROMPT_TEMPLATE.format(
                code_to_opt='\n'.join(code_to_opt),
                suggestion=opt_suggestion
            ),
            'answer': '\n'.join(code_after_opt)
        }


def transform(in_dir, out_dir, fewshot=False):
    assert in_dir.is_dir(), f"Not a directory: {in_dir}"
    assert (in_dir / 'train.json').exists(), f"File train.json does not exist in: {in_dir}"
    assert (in_dir / 'test.json').exists(), f"File test.json does not exist in: {in_dir}"

    out_dir.mkdir(exist_ok=True)

    for fname in ['train.json', 'test.json']:
        with (in_dir / fname).open('r') as fin:
            tx_data = [transform_item(item, fewshot=fewshot) for item in json.load(fin)]
        with (out_dir / fname).open('w') as fou:
            json.dump(tx_data, fou, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(
        prog="tx_coderef",
        description="Tranform the HQCM dataset for finetuning of code refinement"
    )
    parser.add_argument(
        "dataset", type=Path,
        help="Path to the directory saving the HQCM dataset before transformation"
    )
    parser.add_argument(
        "-o", "--output",
        required=True, type=Path,
        help="Path to the directory to save the HQCM dataset after transforming for code refinement"
    )
    parser.add_argument(
        "--fewshot", default=False, action='store_true',
        help="Transform the dataset that can be used for fewshot in-context learning"
    )
    args = parser.parse_args()

    transform(args.dataset, args.output, fewshot=args.fewshot)
