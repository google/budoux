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
"""Deterministic model retraining meta-pipeline orchestrator.

Implements deterministic 80:10:10 KNBC corpus split, separated weighted
encoding (KNBC: scale 1, curated data: scale 100), feature
conflict resolution (-t 0.8), AdaBoost training with 200,000 iterations
(--feature-thres 2), compact JSON export, and holdout/regression
comparative evaluation reports.
"""

import argparse
import glob
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import build_model  # noqa: E402
from scripts import encode_data  # noqa: E402
from scripts import evaluate_model  # noqa: E402
from scripts import find_conflicts  # noqa: E402
from scripts import train  # noqa: E402


def deterministic_split_corpus(
    source_path: str,
    output_dir: str,
    lang: str = "ja",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[str, str, str]:
  """Splits corpus deterministically (shuffled with seed) into 80:10:10 subsets.

  Args:
    source_path: Path to the source corpus file.
    output_dir: Destination directory for split files.
    lang: Language code.
    train_ratio: Proportion of training samples (default 0.80).
    val_ratio: Proportion of validation samples (default 0.10).
    test_ratio: Proportion of test samples (default 0.10).
    seed: Random seed for reproducible shuffling (default 42).

  Returns:
    Tuple of paths to (train_path, val_path, test_path).
  """
  os.makedirs(output_dir, exist_ok=True)
  train_filename = f"{lang}_train.txt" if lang != "ja" else "knbc_train.txt"
  val_filename = f"{lang}_val.txt" if lang != "ja" else "knbc_val.txt"
  test_filename = f"{lang}_test.txt" if lang != "ja" else "knbc_test.txt"

  train_path = os.path.join(output_dir, train_filename)
  val_path = os.path.join(output_dir, val_filename)
  test_path = os.path.join(output_dir, test_filename)

  with open(source_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

  rng = random.Random(seed)
  rng.shuffle(lines)

  total = len(lines)
  n_train = int(total * train_ratio)
  n_val = int(total * val_ratio)

  train_lines = lines[:n_train]
  val_lines = lines[n_train:n_train + n_val]
  test_lines = lines[n_train + n_val:]

  for path, split_lines in [
      (train_path, train_lines),
      (val_path, val_lines),
      (test_path, test_lines),
  ]:
    with open(path, "w", encoding="utf-8") as f:
      for line in split_lines:
        f.write(line + "\n")

  print(f"[Split Corpus] Split {total} sentences deterministically "
        f"(Train: {len(train_lines)}, Val: {len(val_lines)}, "
        f"Test: {len(test_lines)}) into {output_dir}.")
  return train_path, val_path, test_path


def ensure_base_datasets(
    lang: str = "ja",
    split_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  """Ensures baseline corpus 80:10:10 split files exist for the target language."""
  if not split_dir:
    split_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tmp", "splits"))

  train_filename = f"{lang}_train.txt" if lang != "ja" else "knbc_train.txt"
  val_filename = f"{lang}_val.txt" if lang != "ja" else "knbc_val.txt"
  test_filename = f"{lang}_test.txt" if lang != "ja" else "knbc_test.txt"

  train_path = os.path.join(split_dir, train_filename)
  val_path = os.path.join(split_dir, val_filename)
  test_path = os.path.join(split_dir, test_filename)

  if all(os.path.exists(p) for p in (train_path, val_path, test_path)):
    return train_path, val_path, test_path

  if lang == "ja":
    print("[Base Corpus] KNBC 80:10:10 dataset splits not found. Preparing...")
    import tarfile
    import urllib.request

    from scripts import prepare_knbc  # noqa: E402

    tmp_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tmp"))
    os.makedirs(tmp_dir, exist_ok=True)
    knbc_dir = os.path.join(tmp_dir, "KNBC_v1.0_090925_utf8")

    if not os.path.exists(knbc_dir):
      tar_path = os.path.join(tmp_dir, "knbc.tar.bz2")
      if not os.path.exists(tar_path):
        url = (
            "https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2"
        )
        print(f"[Base Corpus] Downloading raw KNBC corpus from {url}...")
        urllib.request.urlretrieve(url, tar_path)

      print("[Base Corpus] Extracting KNBC archive...")
      with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=tmp_dir)

    source_knbc_path = os.path.join(tmp_dir, "source_knbc.txt")
    prepare_knbc.process_knbc(
        source_dir=knbc_dir,
        outfile=source_knbc_path,
    )

    if os.path.exists(source_knbc_path):
      return deterministic_split_corpus(
          source_path=source_knbc_path,
          output_dir=split_dir,
          lang=lang,
          train_ratio=0.80,
          val_ratio=0.10,
          test_ratio=0.10,
      )

  return None, None, None


def merge_and_weight_training_sources(
    lang: str,
    output_path: str,
    base_train_path: Optional[str] = None,
    base_scale: int = 1,
    curated_scale: int = 100,
) -> str:
  """Separated weighted merge: base corpus (scale 1) and curated data (scale 100)."""
  combined: List[str] = []

  # 1. Base KNBC corpus lines repeated base_scale times
  if base_train_path and os.path.exists(base_train_path):
    with open(base_train_path, "r", encoding="utf-8") as f:
      base_lines = [line.strip() for line in f if line.strip()]
      for _ in range(base_scale):
        combined.extend(base_lines)

  # 2. Curated & reviewed samples repeated curated_scale times
  lang_dir = os.path.join(
      os.path.dirname(__file__), "..", "data", "finetuning", lang)
  if os.path.exists(lang_dir):
    for curated_file in sorted(glob.glob(os.path.join(lang_dir, "*.txt"))):
      with open(curated_file, "r", encoding="utf-8") as f:
        curated_lines = [line.strip() for line in f if line.strip()]
        for _ in range(curated_scale):
          combined.extend(curated_lines)

  with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(combined) + "\n")

  print(f"[Merge & Weight] Generated separated weighted training dataset "
        f"({len(combined)} total lines) at {output_path}.")
  return output_path


def run_encode_step(source_path: str, encoded_out_path: str) -> None:
  """Runs feature encoding."""
  encode_data.main([source_path, "-o", encoded_out_path])


def run_find_conflicts_step(encoded_path: str,
                            cleaned_path: str,
                            threshold: float = 0.8) -> None:
  """Resolves feature conflicts at the specified threshold."""
  find_conflicts.find_conflicts(
      data_path=encoded_path, output_path=cleaned_path, threshold=threshold)


def run_train_step(
    cleaned_encoded_path: str,
    weights_out_path: str,
    log_out_path: str,
    iterations: int = 200000,
    feature_thres: int = 2,
    val_data_path: Optional[str] = None,
) -> None:
  """Trains AdaBoost weights using JAX."""
  val_encoded_path: Optional[str] = None
  if val_data_path and os.path.exists(val_data_path):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
      val_encoded_path = tmp.name
    run_encode_step(val_data_path, val_encoded_path)

  try:
    dataset_train, features, dataset_val = train.preprocess(
        train_data_path=cleaned_encoded_path,
        feature_thres=feature_thres,
        val_data_path=val_encoded_path,
    )
    out_span = max(1, iterations // 10)
    train.fit(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        features=features,
        iters=iterations,
        weights_filename=weights_out_path,
        log_filename=log_out_path,
        out_span=out_span,
    )
  finally:
    if val_encoded_path and os.path.exists(val_encoded_path):
      os.remove(val_encoded_path)


def run_build_compact_model_step(
    weights_path: str,
    model_json_out_path: str,
    scale: int = 1000,
) -> Dict[str, Dict[str, int]]:
  """Compiles weights into a compact JSON model representation."""
  with open(weights_path, "r", encoding="utf-8") as f:
    weight_lines = f.readlines()

  aggregated = build_model.aggregate_scores(weight_lines)
  rounded = build_model.round_model(aggregated, scale=scale)

  os.makedirs(
      os.path.dirname(os.path.abspath(model_json_out_path)), exist_ok=True)
  with open(model_json_out_path, "w", encoding="utf-8") as f:
    json.dump(rounded, f, ensure_ascii=False, separators=(",", ":"))

  print(f"[Build Model] Exported compact model JSON to {model_json_out_path}.")
  return rounded


def run_evaluation_report(
    model_path: str,
    holdout_test_path: Optional[str] = None,
    regression_tsv_path: Optional[str] = None,
    baseline_model_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Runs holdout and regression evaluation benchmarks and logs comparative reports."""
  reports: Dict[str, Any] = {}

  if holdout_test_path and os.path.exists(holdout_test_path):
    print("\n=== Holdout Benchmark Report (Generalization) ===")
    h_metrics = evaluate_model.evaluate(model_path, holdout_test_path)
    model_bytes = os.path.getsize(model_path)
    h_report: Dict[str, Any] = dict(h_metrics)
    h_report["model_file_bytes"] = model_bytes
    h_report["model_file_kb"] = round(model_bytes / 1024.0, 2)
    if baseline_model_path and os.path.exists(baseline_model_path):
      b_metrics = evaluate_model.evaluate(baseline_model_path,
                                          holdout_test_path)
      b_bytes = os.path.getsize(baseline_model_path)
      deltas = {
          k: round(h_metrics[k] - b_metrics[k], 6)
          for k in ("accuracy", "precision", "recall", "fscore")
      }
      deltas["model_file_bytes"] = model_bytes - b_bytes
      h_report["baseline"] = dict(b_metrics)
      h_report["baseline"]["model_file_bytes"] = b_bytes
      h_report["baseline"]["model_file_kb"] = round(b_bytes / 1024.0, 2)
      h_report["deltas"] = deltas
    reports["holdout"] = h_report
    print(json.dumps(h_report, indent=2))

  if regression_tsv_path and os.path.exists(regression_tsv_path):
    print("\n=== Regression Suite Report (Quality TSV) ===")
    r_metrics = evaluate_model.evaluate(model_path, regression_tsv_path)
    r_report: Dict[str, Any] = dict(r_metrics)
    if baseline_model_path and os.path.exists(baseline_model_path):
      b_r_metrics = evaluate_model.evaluate(baseline_model_path,
                                            regression_tsv_path)
      r_report["baseline"] = dict(b_r_metrics)
      r_report["deltas"] = {
          k: round(r_metrics[k] - b_r_metrics[k], 6)
          for k in ("accuracy", "precision", "recall", "fscore")
      }
    reports["regression"] = r_report
    print(json.dumps(r_report, indent=2))

  return reports


def run_retraining_pipeline(
    lang: str = "ja",
    iterations: int = 200000,
    feature_thres: int = 2,
    conflict_threshold: float = 0.8,
    base_scale: int = 1,
    curated_scale: int = 100,
    output_model: Optional[str] = None,
    min_accuracy: float = 0.0,
    force: bool = False,
    split_dir: Optional[str] = None,
    regression_tsv_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Runs deterministic model retraining meta-pipeline."""
  if not output_model:
    output_model = os.path.join(
        os.path.dirname(__file__), "..", "budoux", "models", f"{lang}.json")
  baseline_model_path = output_model if os.path.exists(output_model) else None

  if not regression_tsv_path:
    regression_tsv_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "quality", f"{lang}.tsv")

  base_train, base_val, base_test = ensure_base_datasets(
      lang=lang, split_dir=split_dir)

  with tempfile.TemporaryDirectory() as tmp_dir:
    merged_path = os.path.join(tmp_dir, "weighted_train.txt")
    merge_and_weight_training_sources(
        lang=lang,
        output_path=merged_path,
        base_train_path=base_train,
        base_scale=base_scale,
        curated_scale=curated_scale,
    )

    encoded_path = os.path.join(tmp_dir, "encoded.txt")
    run_encode_step(merged_path, encoded_path)

    cleaned_path = os.path.join(tmp_dir, "cleaned.txt")
    run_find_conflicts_step(
        encoded_path, cleaned_path, threshold=conflict_threshold)

    weights_path = os.path.join(tmp_dir, "weights.txt")
    log_path = os.path.join(tmp_dir, "train.log")
    run_train_step(
        cleaned_encoded_path=cleaned_path,
        weights_out_path=weights_path,
        log_out_path=log_path,
        iterations=iterations,
        feature_thres=feature_thres,
        val_data_path=base_val,
    )

    temp_model_path = os.path.join(tmp_dir, "temp_model.json")
    run_build_compact_model_step(
        weights_path=weights_path, model_json_out_path=temp_model_path)

    reports = run_evaluation_report(
        model_path=temp_model_path,
        holdout_test_path=base_test,
        regression_tsv_path=regression_tsv_path,
        baseline_model_path=baseline_model_path,
    )

    # Regression check gate
    regressed = False
    for suite_key, m in reports.items():
      acc = m.get("accuracy", 0.0)
      if acc < min_accuracy:
        regressed = True
      if "delta_accuracy" in m and m["delta_accuracy"] < 0.0:
        regressed = True

    if regressed and not force:
      print(
          "[Error] Regression detected against baseline or minimum accuracy threshold."
      )
      return {
          "status": "regressed",
          "reports": reports,
      }

    # Promote compact model to target output path
    with open(temp_model_path, "r", encoding="utf-8") as f:
      compact_content = f.read()
    with open(output_model, "w", encoding="utf-8") as f:
      f.write(compact_content)

    print(f"\n[Success] Updated production model at {output_model}.")
    return {
        "status": "completed",
        "output_model": output_model,
        "reports": reports,
    }


def build_parser() -> argparse.ArgumentParser:
  """Constructs CLI parser."""
  p = argparse.ArgumentParser(
      description="Deterministic model retraining meta-pipeline orchestrator.")
  p.add_argument(
      "--lang",
      type=str,
      default="ja",
      help="Target language code (default: ja)")
  p.add_argument(
      "--iter",
      type=int,
      default=200000,
      help="Training iterations (default: 200000)",
  )
  p.add_argument(
      "--feature-thres",
      type=int,
      default=2,
      help="Feature frequency threshold (default: 2)",
  )
  p.add_argument(
      "--conflict-threshold",
      type=float,
      default=0.8,
      help="Conflict resolution threshold (default: 0.8)",
  )
  p.add_argument(
      "--base-scale",
      type=int,
      default=1,
      help="Encoding weight scale for base corpus (default: 1)",
  )
  p.add_argument(
      "--curated-scale",
      type=int,
      default=100,
      help="Encoding weight scale for curated data (default: 100)",
  )
  p.add_argument(
      "--output-model", type=str, default=None, help="Destination model path.")
  p.add_argument(
      "--min-accuracy",
      type=float,
      default=0.0,
      help="Minimum required accuracy threshold.",
  )
  p.add_argument(
      "--force",
      action="store_true",
      help="Force model output export even if regressed.",
  )
  return p


def main() -> None:
  """CLI entry point."""
  parser = build_parser()
  args = parser.parse_args()

  res = run_retraining_pipeline(
      lang=args.lang,
      iterations=args.iter,
      feature_thres=args.feature_thres,
      conflict_threshold=args.conflict_threshold,
      base_scale=args.base_scale,
      curated_scale=args.curated_scale,
      output_model=args.output_model,
      min_accuracy=args.min_accuracy,
      force=args.force,
  )

  if res.get("status") == "regressed":
    sys.exit(1)


if __name__ == "__main__":
  main()
