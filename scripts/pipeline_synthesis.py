# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data synthesis meta-pipeline orchestrator.

Generates synthetic training samples from issues or direct text inputs using
synthesize_samples, enforces an interactive human review gate, persists
approved datasets in curated storage, and appends samples to quality suites
(ja.tsv) and curated history (history.txt).
"""

import argparse
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

# Module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import budoux  # noqa: E402
from scripts import synthesize_samples  # noqa: E402

SEP = budoux.utils.SEP  # Canonical character separator '▁'


def human_sample_review_gate(
    synthesized_lines: List[str],
    auto_accept: bool = False,
    input_func: Callable[[str], str] = input,
) -> List[str]:
  """Human Sample Review Gate: Interactively reviews and approves candidate lines.

  Args:
    synthesized_lines: Synthesized candidate sample sentences.
    auto_accept: If True, automatically accepts all lines without prompt.
    input_func: Function to read user input (injectable for testing).

  Returns:
    List[str]: Approved (and potentially edited) training sample lines.
  """
  if not synthesized_lines:
    print("[Human Review Gate] No synthesized lines provided for review.")
    return []

  if auto_accept:
    print("[Human Review Gate] Auto-accept enabled. Approving all lines.")
    return list(synthesized_lines)

  print("\n=== Human Sample Review Gate ===")
  for idx, line in enumerate(synthesized_lines, 1):
    print(f"  [{idx}] {line}")

  while True:
    choice = (
        input_func("\nApprove synthesized samples? ([Y]es / [e]dit / [n]o): ")
        .strip().lower())
    if choice in ("", "y", "yes"):
      print("[Human Review Gate] Samples approved.")
      return list(synthesized_lines)
    elif choice in ("n", "no"):
      print("[Human Review Gate] Samples rejected by maintainer.")
      return []
    elif choice in ("e", "edit"):
      edited_lines = list(synthesized_lines)
      while True:
        print("\nCurrent candidate lines:")
        for idx, line in enumerate(edited_lines, 1):
          print(f"  [{idx}] {line}")
        action = (
            input_func(
                "\nEnter line number to edit/delete, 'add' to insert, or 'done' to finish: "
            ).strip().lower())
        if action == "done":
          print("[Human Review Gate] Editing complete. Approved edited lines.")
          return edited_lines
        elif action == "add":
          new_line = input_func("Enter new sample sentence with '▁': ").strip()
          if new_line:
            edited_lines.append(new_line)
        elif action.isdigit():
          line_idx = int(action) - 1
          if 0 <= line_idx < len(edited_lines):
            print(f"Selected [{line_idx + 1}]: {edited_lines[line_idx]}")
            sub_action = (
                input_func("Choose action ([e]dit / [d]elete / [c]ancel): ")
                .strip().lower())
            if sub_action in ("e", "edit"):
              val = input_func("Enter replacement sentence: ").strip()
              if val:
                edited_lines[line_idx] = val
            elif sub_action in ("d", "delete"):
              edited_lines.pop(line_idx)
          else:
            print("Invalid line index.")
    else:
      print("Invalid choice. Please enter 'y', 'e', or 'n'.")


def register_synthetic_dataset(
    lang: str,
    issue_id: str,
    approved_lines: List[str],
) -> str:
  """Persists approved samples into language dataset directory.

  Args:
    lang: Target language code.
    issue_id: Issue ID (e.g. '468').
    approved_lines: Approved synthetic lines.

  Returns:
    str: Path to written issue file in language directory.
  """
  lang_dir = os.path.join(
      os.path.dirname(__file__), "..", "data", "finetuning", lang)
  os.makedirs(lang_dir, exist_ok=True)

  issue_filename = f"issue_{issue_id}.txt"
  issue_path = os.path.join(lang_dir, issue_filename)
  with open(issue_path, "w", encoding="utf-8") as f:
    f.write("\n".join(approved_lines) + "\n")

  print(f"[Dataset Storage] Saved dataset to {issue_path}.")
  return issue_path


def append_to_quality_suite(
    lang: str,
    issue_id: str,
    sample_line: str,
    quality_path: Optional[str] = None,
) -> None:
  """Appends a representative fix line to tests/quality/{lang}.tsv.

  Args:
    lang: Language code.
    issue_id: Issue ID.
    sample_line: Sample sentence with separators '▁'.
    quality_path: Optional path to quality tsv file.
  """
  if not quality_path:
    quality_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "quality", f"{lang}.tsv")
  if not os.path.exists(quality_path):
    return

  label = f"gh{issue_id}"
  formatted_entry = f"{label}\t{sample_line}\n"

  with open(quality_path, "r", encoding="utf-8") as f:
    content = f.read()

  if formatted_entry.strip() not in content:
    with open(quality_path, "a", encoding="utf-8") as f:
      f.write(formatted_entry)
    print(
        f"[Test Suite] Registered representative test case into {quality_path}."
    )


def append_to_history(
    lang: str,
    approved_lines: List[str],
    history_path: Optional[str] = None,
) -> None:
  """Appends approved sample lines to curated history.txt.

  Args:
    lang: Target language code.
    approved_lines: List of approved sample sentences.
    history_path: Optional path to history file.
  """
  if not history_path:
    history_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "finetuning",
        lang,
        "history.txt",
    )
  os.makedirs(os.path.dirname(history_path), exist_ok=True)

  existing_lines = set()
  if os.path.exists(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
      existing_lines = {line.strip() for line in f if line.strip()}

  new_lines = [
      item.strip()
      for item in approved_lines
      if item.strip() and item.strip() not in existing_lines
  ]

  if new_lines:
    with open(history_path, "a", encoding="utf-8") as f:
      for line in new_lines:
        f.write(line + "\n")
    print(f"[History] Appended {len(new_lines)} lines to {history_path}.")


def run_synthesis_pipeline(
    inputs: Optional[List[str]] = None,
    issues: Optional[List[str]] = None,
    lang: str = "ja",
    num_candidates: int = 30,
    max_keep: int = 15,
    auto_accept: bool = False,
    append_history: bool = True,
    client: Optional[Any] = None,
    parser: Optional[budoux.Parser] = None,
    input_func: Callable[[str], str] = input,
    quality_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Runs the end-to-end data synthesis pipeline.

  Args:
    inputs: Target input strings.
    issues: Target GitHub issue IDs.
    lang: Language code.
    num_candidates: Candidates per target.
    max_keep: Max candidates to retain per target.
    auto_accept: Bypass interactive prompt.
    append_history: Whether to append approved lines to history.txt.
    client: Optional genai.Client instance.
    parser: Optional budoux.Parser instance.
    input_func: Callable for reading user prompts.
    quality_path: Optional path to quality TSV.
    history_path: Optional path to history TXT.

  Returns:
    Dict containing synthesis status, approved sample count, and saved paths.
  """
  targets_to_process: List[Tuple[Optional[str], Optional[str]]] = []
  if inputs:
    for item_inp in inputs:
      targets_to_process.append((item_inp, None))
  if issues:
    for item_iss in issues:
      targets_to_process.append((None, item_iss))

  if not targets_to_process:
    print("[Synthesis Pipeline] No target inputs or issues provided.")
    return {"status": "aborted", "reason": "No target inputs or issues."}

  all_approved_lines: List[str] = []
  saved_paths: List[str] = []

  with tempfile.TemporaryDirectory() as tmp_dir:
    for idx, (raw_inp, raw_iss) in enumerate(targets_to_process):
      target_id: str = (
          raw_iss if raw_iss is not None else f"direct_input_{idx + 1}")
      tmp_staging = os.path.join(tmp_dir, f"staging_{target_id}.txt")
      lines = synthesize_samples.run_agentic_synthesis_pipeline(
          input_str=raw_inp,
          issue_id=raw_iss,
          num_candidates=num_candidates,
          max_keep=max_keep,
          lang=lang,
          outfile=tmp_staging,
          client=client,
          parser=parser,
      )
      if not lines:
        continue

      print(
          f"\n[Synthesis Pipeline] Reviewing candidates for target: {target_id}"
      )
      target_approved = human_sample_review_gate(
          synthesized_lines=lines,
          auto_accept=auto_accept,
          input_func=input_func,
      )
      if target_approved:
        all_approved_lines.extend(target_approved)
        issue_path = register_synthetic_dataset(
            lang=lang,
            issue_id=target_id,
            approved_lines=target_approved,
        )
        saved_paths.append(issue_path)
        append_to_quality_suite(
            lang=lang,
            issue_id=target_id,
            sample_line=target_approved[0],
            quality_path=quality_path,
        )

    if not all_approved_lines:
      print("[Synthesis Pipeline] Review rejected or no samples approved.")
      return {
          "status": "aborted",
          "reason": "Human Review Gate rejected samples.",
      }

    if append_history:
      append_to_history(
          lang=lang,
          approved_lines=all_approved_lines,
          history_path=history_path,
      )

    return {
        "status": "completed",
        "approved_lines": all_approved_lines,
        "approved_lines_count": len(all_approved_lines),
        "saved_paths": saved_paths,
    }


def build_parser() -> argparse.ArgumentParser:
  """Constructs CLI parser."""
  p = argparse.ArgumentParser(
      description="Data synthesis meta-pipeline orchestrator.")
  group = p.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--input",
      "-i",
      type=str,
      help="Target string or comma-separated strings.",
  )
  group.add_argument(
      "--issue",
      type=str,
      help="GitHub issue ID or comma-separated issue IDs.",
  )
  p.add_argument(
      "--lang",
      type=str,
      default="ja",
      help="Target language code (default: ja)",
  )
  p.add_argument(
      "--num-candidates",
      type=int,
      default=30,
      help="Candidates per target.",
  )
  p.add_argument(
      "--max-keep", type=int, default=15, help="Max candidates per target.")
  p.add_argument(
      "--auto-accept",
      action="store_true",
      help="Bypass interactive review prompt.",
  )
  p.add_argument(
      "--no-history",
      action="store_true",
      help="Do not append approved lines to curated history.txt.",
  )
  return p


def main() -> None:
  """CLI entry point."""
  parser = build_parser()
  args = parser.parse_args()

  inputs = [x.strip() for x in args.input.split(",")] if args.input else None
  issues = [x.strip() for x in args.issue.split(",")] if args.issue else None

  res = run_synthesis_pipeline(
      inputs=inputs,
      issues=issues,
      lang=args.lang,
      num_candidates=args.num_candidates,
      max_keep=args.max_keep,
      auto_accept=args.auto_accept,
      append_history=not args.no_history,
  )

  if res.get("status") == "aborted":
    sys.exit(1)


if __name__ == "__main__":
  main()
