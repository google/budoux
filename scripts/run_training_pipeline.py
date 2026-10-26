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
"""Streamlined retraining meta-pipeline for BudouX models."""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, LIB_PATH)

from scripts import colab_runner, encode_data

KNBC_URL = "https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2"
MAX_AUTO_WEIGHT_SCALE = 30


def parse_model_id(model_id: str) -> tuple[str, str]:
  """Splits a model identifier into its BCP-47 language tag and optional flavor.

  Model identifiers follow the `<bcp47_lang>` or `<bcp47_lang>_<flavor>`
  convention (e.g. 'ja' -> ('ja', ''), 'ja_word' -> ('ja', 'word'),
  'zh-hans' -> ('zh-hans', ''), 'zh-hans_word' -> ('zh-hans', 'word')).

  Args:
    model_id: Model identifier string.

  Returns:
    A tuple of (base_lang, flavor).
  """
  base_lang, _, flavor = model_id.partition("_")
  return base_lang, flavor


def get_split_prefix(lang: str) -> str:
  """Returns the dataset file prefix for a given model identifier."""
  base_lang, flavor = parse_model_id(lang)
  # The default Japanese phrase model ('ja') is trained on the KNBC corpus,
  # whereas word-level ('ja_word') and other models ('zh-hans', 'zh-hant', etc.)
  # use the Unicode Test Corpora.
  if base_lang == "ja" and not flavor:
    return "knbc"
  return f"unicode_{lang}"


def _build_train_cmd(
  *,
  executable: list[str],
  script_path: str,
  train_data: str,
  weights_path: str,
  log_path: str,
  iterations: int,
  feature_thres: int,
  algorithm: str,
  val_data: str | None = None,
  patience: int | None = None,
  learning_rate: float | None = None,
) -> list[str]:
  """Builds the CLI command list to invoke `scripts/train.py`."""
  cmd = [
    *executable,
    script_path,
    train_data,
    "-o",
    weights_path,
    "--log",
    log_path,
    "--iter",
    str(iterations),
    "--feature-thres",
    str(feature_thres),
    "--algorithm",
    algorithm,
  ]
  if val_data is not None:
    cmd.extend(["--val-data", val_data])
  if patience is not None:
    cmd.extend(["--patience", str(patience)])
  if learning_rate is not None:
    cmd.extend(["--learning-rate", str(learning_rate)])
  return cmd


def run_retraining_pipeline(
  lang: str = "ja",
  iterations: int = 10000,
  feature_thres: int = 2,
  split_dir: str = "tmp/splits",
  out_model: str = "",
  weight_factor: int | None = None,
  patience: int | None = None,
  algorithm: str = "logitboost",
  learning_rate: float | None = None,
  colab: bool = False,
  accelerator: str = "T4",
  session_name: str | None = None,
) -> None:
  """Executes end-to-end dataset preparation, JAX fitting, and validation."""

  if not out_model:
    out_model = os.path.join("budoux", "models", f"{lang}.json")

  base_lang, flavor = parse_model_id(lang)
  prefix = get_split_prefix(lang)
  train_file = os.path.join(split_dir, f"{prefix}_train.txt")
  val_split = os.path.join(split_dir, f"{prefix}_val.txt")
  test_split = os.path.join(split_dir, f"{prefix}_test.txt")

  if os.path.exists(train_file):
    print(f"[Dataset] Using existing {prefix} dataset splits in {split_dir}.")
  elif base_lang == "ja" and not flavor:
    print(f"[Dataset] Fetching KNBC corpus from {KNBC_URL}...")
    with tempfile.TemporaryDirectory() as tmp_knbc:
      tar_path = os.path.join(tmp_knbc, "knbc.tar.bz2")
      urllib.request.urlretrieve(KNBC_URL, tar_path)
      with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=tmp_knbc, filter="data")
      source_dir = os.path.join(tmp_knbc, "KNBC_v1.0_090925_utf8")
      outfile = os.path.join(tmp_knbc, "source_knbc.txt")
      subprocess.run(
        [
          sys.executable,
          "scripts/prepare_knbc.py",
          source_dir,
          "-o",
          outfile,
          "--split-dir",
          split_dir,
        ],
        check=True,
      )
  else:
    print(f"[Dataset] Preparing Unicode corpus splits for '{lang}' in {split_dir}...")
    with tempfile.TemporaryDirectory() as tmp_unicode:
      outfile = os.path.join(tmp_unicode, f"source_{prefix}.txt")
      prep_cmd = [
        sys.executable,
        "scripts/prepare_unicode.py",
        lang,
        "-o",
        outfile,
        "--split-dir",
        split_dir,
      ]
      if flavor != "word":
        prep_cmd.append("--line-break")
      subprocess.run(prep_cmd, check=True)

  with tempfile.TemporaryDirectory() as tmp:
    finetuning_pattern = os.path.join("data", "finetuning", lang, "*.txt")
    ft_files = sorted(glob.glob(finetuning_pattern))

    scale = 1
    if weight_factor is not None:
      scale = weight_factor
      print(f"[Scale] Using specified fine-tuning scale: {scale}x")
    elif ft_files and os.path.exists(train_file):
      with open(train_file, encoding="utf-8") as in_f:
        base_s, _ = encode_data.normalize_input(in_f.read())
      ft_text = ""
      for ft_file in ft_files:
        with open(ft_file, encoding="utf-8") as in_f:
          ft_text += in_f.read()
      ft_s, _ = encode_data.normalize_input(ft_text)
      n_base = len(base_s)
      n_ft = len(ft_s)
      # Equalize total sample weight mass between the large base corpus
      # (n_base * 1) and the small fine-tuning corpus (n_ft * scale) so that
      # targeted fine-tuning boundaries are not drowned out during conflict
      # filtering and boosting, while capping at MAX_AUTO_WEIGHT_SCALE so tiny
      # initial fine-tuning files do not overpower general corpus statistics.
      scale = (
        min(MAX_AUTO_WEIGHT_SCALE, max(1, round(n_base / n_ft))) if n_ft > 0 else 1
      )
      print(
        f"[Scale] Auto-balanced fine-tuning scale: {scale}x (base={n_base}, ft={n_ft})"
      )

    encoded_train = os.path.join(tmp, "encoded.txt")
    base_encoded = os.path.join(tmp, "base_encoded.txt")
    ft_encoded = os.path.join(tmp, "ft_encoded.txt")
    cleaned_train = os.path.join(tmp, "cleaned.txt")
    val_encoded = os.path.join(tmp, "val_encoded.txt")
    weights_path = os.path.join(tmp, "weights.txt")
    log_path = os.path.join(tmp, "train.log")

    print("[Encode] Encoding feature boundaries...")
    if os.path.exists(train_file):
      subprocess.run(
        [
          sys.executable,
          "scripts/encode_data.py",
          train_file,
          "-o",
          base_encoded,
          "--scale",
          "1",
        ],
        check=True,
      )

    if ft_files:
      ft_raw = os.path.join(tmp, "ft_raw.txt")
      with open(ft_raw, "w", encoding="utf-8") as out_f:
        for ft_file in ft_files:
          with open(ft_file, encoding="utf-8") as in_f:
            shutil.copyfileobj(in_f, out_f)
      subprocess.run(
        [
          sys.executable,
          "scripts/encode_data.py",
          ft_raw,
          "-o",
          ft_encoded,
          "--scale",
          str(scale),
        ],
        check=True,
      )

    with open(encoded_train, "w", encoding="utf-8") as out_f:
      if os.path.exists(base_encoded):
        with open(base_encoded, encoding="utf-8") as in_f:
          shutil.copyfileobj(in_f, out_f)
      if os.path.exists(ft_encoded):
        with open(ft_encoded, encoding="utf-8") as in_f:
          shutil.copyfileobj(in_f, out_f)

    print("[Filter] Resolving conflict boundaries...")
    subprocess.run(
      [
        sys.executable,
        "scripts/find_conflicts.py",
        encoded_train,
        "-o",
        cleaned_train,
        "-t",
        "0.8",
      ],
      check=True,
    )

    if os.path.exists(val_split):
      subprocess.run(
        [sys.executable, "scripts/encode_data.py", val_split, "-o", val_encoded],
        check=True,
      )

    has_val = os.path.exists(val_encoded)

    print(f"[Train] Fitting JAX model ({iterations} iterations)...")
    if colab:
      sess = session_name or f"budoux-train-{accelerator}"
      with colab_runner.ColabRunner(
        session_name=sess, accelerator=accelerator
      ) as runner:
        runner.upload_file(cleaned_train, "/content/cleaned.txt")
        runner.upload_file(
          os.path.join(LIB_PATH, "scripts", "train.py"), "/content/train.py"
        )
        if has_val:
          runner.upload_file(val_encoded, "/content/val_encoded.txt")

        remote_args = _build_train_cmd(
          executable=["python3", "-u"],
          script_path="/content/train.py",
          train_data="/content/cleaned.txt",
          weights_path="/content/weights.txt",
          log_path="/content/train.log",
          iterations=iterations,
          feature_thres=feature_thres,
          algorithm=algorithm,
          val_data="/content/val_encoded.txt" if has_val else None,
          patience=patience,
          learning_rate=learning_rate,
        )

        runner.exec_cmd(remote_args)
        runner.download_file("/content/weights.txt", weights_path)

    else:
      train_cmd = _build_train_cmd(
        executable=[sys.executable],
        script_path="scripts/train.py",
        train_data=cleaned_train,
        weights_path=weights_path,
        log_path=log_path,
        iterations=iterations,
        feature_thres=feature_thres,
        algorithm=algorithm,
        val_data=val_encoded if has_val else None,
        patience=patience,
        learning_rate=learning_rate,
      )
      subprocess.run(train_cmd, check=True)

    print(f"[Export] Exporting compact JSON model to {out_model}...")
    build_cmd = [
      sys.executable,
      "scripts/build_model.py",
      weights_path,
      "-o",
      out_model,
    ]
    subprocess.run(build_cmd, check=True)

    print("[Evaluation] Evaluating quality suite benchmark...")
    quality_file = os.path.join("tests", "quality", f"{lang}.tsv")
    if os.path.exists(quality_file):
      subprocess.run(
        [
          sys.executable,
          "-m",
          "budoux.evaluate_model",
          "-m",
          out_model,
          "-t",
          quality_file,
        ],
        check=True,
      )

    if os.path.exists(test_split):
      print("[Evaluation] Evaluating test split benchmark...")
      subprocess.run(
        [
          sys.executable,
          "-m",
          "budoux.evaluate_model",
          "-m",
          out_model,
          "-t",
          test_split,
        ],
        check=True,
      )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Streamlined retraining meta-pipeline for BudouX models."
  )
  parser.add_argument(
    "--lang", type=str, default="ja", help="Language code (default: ja)"
  )
  parser.add_argument(
    "--iter",
    type=int,
    default=10000,
    help="Number of training iterations (default: 10000)",
  )
  parser.add_argument(
    "--feature-thres",
    type=int,
    default=2,
    help="Threshold value of minimum feature frequency (default: 2)",
  )
  parser.add_argument(
    "--split-dir",
    type=str,
    default="tmp/splits",
    help="Directory containing or receiving dataset splits",
  )
  parser.add_argument(
    "--out-model",
    type=str,
    default="",
    help="Output model JSON path (default: budoux/models/<lang>.json)",
  )
  parser.add_argument(
    "--weight-factor",
    type=int,
    default=None,
    help="Oversampling weight scale for fine-tuning datasets (default: auto-balanced based on corpus size ratio)",
  )
  parser.add_argument(
    "--patience",
    type=int,
    default=None,
    help="Number of evaluation spans without improvement before early stopping.",
  )
  parser.add_argument(
    "--algorithm",
    type=str,
    default="logitboost",
    choices=["adaboost", "logitboost"],
    help="Boosting algorithm to use (adaboost or logitboost, default: logitboost)",
  )
  parser.add_argument(
    "--learning-rate",
    "-lr",
    dest="learning_rate",
    type=float,
    default=None,
    help="Learning rate (shrinkage factor) for LogitBoost training.",
  )
  parser.add_argument(
    "--colab", action="store_true", help="Offload JAX training step to remote Colab VM."
  )
  parser.add_argument(
    "--accelerator",
    type=str,
    default="T4",
    help="Accelerator type for Colab VM (default: T4)",
  )

  parser.add_argument(
    "--session-name", type=str, default=None, help="Remote Colab session identifier"
  )
  args = parser.parse_args()
  run_retraining_pipeline(
    lang=args.lang,
    iterations=args.iter,
    feature_thres=args.feature_thres,
    split_dir=args.split_dir,
    out_model=args.out_model,
    weight_factor=args.weight_factor,
    patience=args.patience,
    algorithm=args.algorithm,
    learning_rate=args.learning_rate,
    colab=args.colab,
    accelerator=args.accelerator,
    session_name=args.session_name,
  )


if __name__ == "__main__":
  main()
