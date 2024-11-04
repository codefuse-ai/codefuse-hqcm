import functools
import json

from unidiff import PatchSet

CHSUM_PROMPT_TEMPLATE = """\
Please generate a commit message for the following diff:

```diff
{diff}
```
"""

CHCL_PROMPT_TEMPLATE = """\
A git commit can typically be classified into specific categories by examining its code changes and commit message. These categories include:

- "style": Changes that solely improve the code's formatting and appearance without affecting functionality (e.g., adjusting whitespace, fixing indentation, cleaning up code formatting).
- "docs": Updates or improvements to documentation, which may include inline code comments, README files, or any other type of documentation associated with the project.
- "test": Modifications exclusively related to test code, like the addition of new tests or the correction and improvement of existing tests.
- "build": Changes that affect the build system or tools (like Gulp, Broccoli, NPM) or alterations to external dependencies (e.g., library or package updates).
- "cicd": Tweaks to configuration files or scripts used in Continuous Integration/Continuous Deployment (CI/CD) systems, such as Travis CI or CircleCI configurations.
- "fix": Code amendments that focus on rectifying errors, fixing bugs, or patching security vulnerabilities.
- "feat": Commits that introduce new features or capabilities to the project, such as new classes, functions, or methods.
- "refactor": Changes that reorganize and clean up the codebase without modifying its external behavior or outputs, improving readability and maintainability.

For a given git commit, we can inspect its code difference (diff) and the associated commit message to determine its type. Below is the diff for a specific git commit:

```
{diff}
```

Accompanying this code diff is its commit message:

```
{message}
```

Given this information, the git commit can be categorized as type: """

CHCL_FEWSHOT_PROMPT_TEMPLATE = """\
A git commit can typically be classified into specific categories by examining its code changes and commit message. These categories include:

- "style": Changes that solely improve the code's formatting and appearance without affecting functionality (e.g., adjusting whitespace, fixing indentation, cleaning up code formatting).
- "docs": Updates or improvements to documentation, which may include inline code comments, README files, or any other type of documentation associated with the project.
- "test": Modifications exclusively related to test code, like the addition of new tests or the correction and improvement of existing tests.
- "build": Changes that affect the build system or tools (like Gulp, Broccoli, NPM) or alterations to external dependencies (e.g., library or package updates).
- "cicd": Tweaks to configuration files or scripts used in Continuous Integration/Continuous Deployment (CI/CD) systems, such as Travis CI or CircleCI configurations.
- "fix": Code amendments that focus on rectifying errors, fixing bugs, or patching security vulnerabilities.
- "feat": Commits that introduce new features or capabilities to the project, such as new classes, functions, or methods.
- "refactor": Changes that reorganize and clean up the codebase without modifying its external behavior or outputs, improving readability and maintainability.

For a given git commit, we can inspect its code difference (diff) and the associated commit message to determine its type.

Diff: ```diff
diff --git a/util/src/com/intellij/util/containers/SLRUMap.java b/util/src/com/intellij/util/containers/SLRUMap.java
index 7f3d09c..635dfab 100644
--- a/util/src/com/intellij/util/containers/SLRUMap.java
+++ b/util/src/com/intellij/util/containers/SLRUMap.java
@@ -69,12 +69,12 @@ public class SLRUMap<K,V> {{
   public void put(K key, V value) {{
     V oldValue = myProtectedQueue.remove(key);
     if (oldValue != null) {{
-      onDropFromCache(key, value);
+      onDropFromCache(key, oldValue);
     }}
 
     oldValue = myProbationalQueue.put(getStableKey(key), value);
     if (oldValue != null) {{
-      onDropFromCache(key, value);
+      onDropFromCache(key, oldValue);
     }}
   }}
```
Message: Corrected parameter error in onDropFromCache() function call
Type: fix
Reason: The git commit is a "fix" commit as it rectified a parameter error where `oldValue` should be passed as the argument of `onDropFromCache` rather than `value`.

Diff: ```diff
{diff}
```
Message: {message}
Type: """

CODEREF_PROMPT_TEMPLATE = """\
// Please optimize the given "Code (to optimize)" (a portion of some "File"s) by strictly following the given "Suggestion".
// Your optimization can involve editing or removing existing code, or adding new code.
// You may pay more attention to lines marked by "// !!attention". If no such marked lines, do everything by yourself.

// Code (to optimize):

{code_to_opt}

// Suggestion: {suggestion}

// Code (after optimization):

"""

CODEREF_FEWSHOT_PROMPT_TEMPLATE = """\
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


def xitem_coderef(item, fewshot=False):
    opt_suggestion = item['summaries']['en']

    code_to_opt = []
    code_after_opt = []

    for patched_file in PatchSet(item['change']):
        file_name = patched_file.path
        source_text = "\n".join(
            # Add an attention to additionally inform the model
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
            'prompt': CODEREF_FEWSHOT_PROMPT_TEMPLATE.format(
                code_to_opt='\n'.join(code_to_opt),
                suggestion=opt_suggestion
            ),
            'answer': '\n'.join(code_after_opt)
        }
    else:
        return {
            'prompt': CODEREF_PROMPT_TEMPLATE.format(
                code_to_opt='\n'.join(code_to_opt),
                suggestion=opt_suggestion
            ),
            'answer': '\n'.join(code_after_opt)
        }


def xitem_chcl(item, fewshot=False):
    if fewshot:
        return {
            'prompt': CHCL_FEWSHOT_PROMPT_TEMPLATE.format(
                diff=item['change'],
                message=item['summaries']['en']
            ),
            'answer': item['type']
        }
    else:
        return {
            'prompt': CHCL_PROMPT_TEMPLATE.format(
                diff=item['change'],
                message=item['summaries']['en']
            ),
            'answer': item['type']
        }


def xitem_chsum(item):
    return {
        'prompt': CHSUM_PROMPT_TEMPLATE.format(diff=item['change']),
        'answer': item['summaries']['en']
    }


def transform(in_dir, out_dir, xitem_fn):
    assert in_dir.is_dir(), f"Not a directory: {in_dir}"
    assert (in_dir / 'train.json').exists(), f"File train.json does not exist in: {in_dir}"
    assert (in_dir / 'test.json').exists(), f"File test.json does not exist in: {in_dir}"

    out_dir.mkdir(exist_ok=True)

    for fname in ['train.json', 'test.json']:
        with (in_dir / fname).open('r') as fin:
            tx_data = [xitem_fn(item) for item in json.load(fin)]
        with (out_dir / fname).open('w') as fou:
            json.dump(tx_data, fou, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(
        prog="xdata",
        description="Tranform the HQCM dataset for fine-tuning of a specific code-change task"
    )
    parser.add_argument(
        "dataset", type=Path,
        help="Path to the directory saving the HQCM dataset before transformation"
    )
    parser.add_argument(
        "-t", "--task",
        required=True, choices=['chsum', 'chcl', 'coderef'],
        help="Target tasks: chsum for change summarization, chcl for change classification, and coderef for code refinement"
    )
    parser.add_argument(
        "-o", "--output",
        required=True, type=Path,
        help="Path to the directory to save the HQCM dataset after transforming for the task"
    )
    parser.add_argument(
        "-F", "--few-shot",
        default=False, action='store_true',
        help="Transform the dataset that can be used for few-shot in-context learning"
    )
    args = parser.parse_args()

    if args.task == 'chsum':
        xitem_fn = xitem_chsum
    elif args.task == 'chcl':
        xitem_fn = functools.partial(xitem_chcl, fewshot=args.fewshot)
    elif args.task == 'coderef':
        xitem_fn = functools.partial(xitem_coderef, fewshot=args.fewshot)
    else:
        assert False, "Unsupported code change task: " + args.task

    transform(args.dataset, args.output, xitem_fn=xitem_fn)
