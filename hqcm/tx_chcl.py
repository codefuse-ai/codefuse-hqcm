import json
import sys


PROMPT_TEMPLATE = """\
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


FEWSHOT_PROMPT_TEMPLATE = """\
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


def transform_item(item, fewshot=False):
    if fewshot:
        return {
            'prompt': FEWSHOT_PROMPT_TEMPLATE.format(
                diff=item['change'],
                message=item['summaries']['en']
            ),
            'answer': item['type']
        }
    else:
        return {
            'prompt': PROMPT_TEMPLATE.format(
                diff=item['change'],
                message=item['summaries']['en']
            ),
            'answer': item['type']
        }


def transform(in_dir, out_dir, fewshot=False):
    assert in_dir.is_dir(), f"Not a directory: {in_dir}"
    assert (in_dir / 'train.json').exists(), f"File train.json does not exist in: {in_dir}"
    assert (in_dir / 'test.json').exists(), f"File test.json does not exist in: {in_dir}"

    out_dir.mkdir(exist_ok=True)

    for fname in ['train.json', 'test.json']:
        with (in_dir / fname).open('r') as fin:
            tx_data = [transform_item(item, fewshot) for item in json.load(fin)]
        with (out_dir / fname).open('w') as fou:
            json.dump(tx_data, fou, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(
        prog="tx_chcl",
        description="Tranform the HQCM dataset for finetuning of change classification"
    )
    parser.add_argument(
        "dataset", type=Path,
        help="Path to the directory saving the HQCM dataset before transformation"
    )
    parser.add_argument(
        "-o", "--output",
        required=True, type=Path,
        help="Path to the directory to save the HQCM dataset after transforming for change classification"
    )
    parser.add_argument(
        "--fewshot", default=False, action='store_true',
        help="Transform the dataset that can be used for fewshot in-context learning"
    )
    args = parser.parse_args()

    transform(args.dataset, args.output, fewshot=args.fewshot)
