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

from scripts import colab_runner

KNBC_URL = "https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2"


def run_retraining_pipeline(
  lang: str = "ja",
  iterations: int = 200000,
  feature_thres: int = 2,
  split_dir: str = "tmp/splits",
  out_model: str = "",
  weight_factor: int = 100,
  colab: bool = False,
  accelerator: str = "T4",
  session_name: str | None = None,
) -> None:
  """Executes end-to-end dataset preparation, JAX fitting, and validation."""

  if not out_model:
    out_model = os.path.join("budoux", "models", f"{lang}.json")

  train_file = os.path.join(split_dir, "knbc_train.txt")
  if os.path.exists(train_file):
    print(f"[Dataset] Using existing KNBC dataset splits in {split_dir}.")
  else:
    print(f"[Dataset] Fetching KNBC corpus from {KNBC_URL}...")
    with tempfile.TemporaryDirectory() as tmp_knbc:
      tar_path = os.path.join(tmp_knbc, "knbc.tar.bz2")
      urllib.request.urlretrieve(KNBC_URL, tar_path)
      with tarfile.open(tar_path, "r:bz2") as tar:
        if hasattr(tarfile, "data_filter"):
          tar.extractall(path=tmp_knbc, filter="data")
        else:
          abs_out = os.path.abspath(tmp_knbc)
          for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(abs_out, member.name))
            if os.path.commonpath([abs_out, member_path]) != abs_out:
              raise ValueError(f"Insecure member in tar file: {member.name}")
          tar.extractall(path=tmp_knbc)
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

  with tempfile.TemporaryDirectory() as tmp:
    weighted_train = os.path.join(tmp, "weighted_train.txt")
    finetuning_pattern = os.path.join("data", "finetuning", lang, "*.txt")

    print(
      f"[Merge] Combining fine-tuning data ({finetuning_pattern}) with {weight_factor}x weight..."
    )
    with open(weighted_train, "w", encoding="utf-8") as out_f:
      for ft_file in sorted(glob.glob(finetuning_pattern)):
        with open(ft_file, encoding="utf-8") as in_f:
          content = in_f.read()
        out_f.writelines(content for _ in range(weight_factor))
      knbc_train = os.path.join(split_dir, "knbc_train.txt")
      if os.path.exists(knbc_train):
        with open(knbc_train, encoding="utf-8") as in_f:
          shutil.copyfileobj(in_f, out_f)

    encoded_train = os.path.join(tmp, "encoded.txt")
    cleaned_train = os.path.join(tmp, "cleaned.txt")
    val_encoded = os.path.join(tmp, "val_encoded.txt")
    weights_path = os.path.join(tmp, "weights.txt")

    print("[Encode] Encoding feature boundaries...")
    subprocess.run(
      [sys.executable, "scripts/encode_data.py", weighted_train, "-o", encoded_train],
      check=True,
    )

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

    val_split = os.path.join(split_dir, "knbc_val.txt")
    if os.path.exists(val_split):
      subprocess.run(
        [sys.executable, "scripts/encode_data.py", val_split, "-o", val_encoded],
        check=True,
      )

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
        if os.path.exists(val_encoded):
          runner.upload_file(val_encoded, "/content/val_encoded.txt")

        # Outputs training metrics and weights regularly across total iterations.
        out_span = min(5000, max(1, iterations // 10))
        remote_args = [
          "python3",
          "-u",
          "/content/train.py",
          "/content/cleaned.txt",
          "-o",
          "/content/weights.txt",
          "--log",
          "/content/train.log",
          "--iter",
          str(iterations),
          "--feature-thres",
          str(feature_thres),
          "--out-span",
          str(out_span),
        ]

        if os.path.exists(val_encoded):
          remote_args.extend(["--val-data", "/content/val_encoded.txt"])

        runner.exec_cmd(remote_args)
        runner.download_file("/content/weights.txt", weights_path)

    else:
      train_cmd = [
        sys.executable,
        "scripts/train.py",
        cleaned_train,
        "-o",
        weights_path,
        "--iter",
        str(iterations),
        "--feature-thres",
        str(feature_thres),
      ]
      if os.path.exists(val_encoded):
        train_cmd.extend(["--val-data", val_encoded])
      subprocess.run(train_cmd, check=True)

    print(f"[Export] Exporting compact JSON model to {out_model}...")
    subprocess.run(
      [sys.executable, "scripts/build_model.py", weights_path, "-o", out_model],
      check=True,
    )

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

    test_split = os.path.join(split_dir, "knbc_test.txt")
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
    default=200000,
    help="Number of training iterations (default: 200000)",
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
    help="Directory containing or receiving KNBC splits",
  )
  parser.add_argument(
    "--out-model",
    type=str,
    default="",
    help="Output model JSON path (default: budoux/models/<lang>.json)",
  )
  parser.add_argument(
    "--colab",
    action="store_true",
    help="Offload JAX AdaBoost training step to remote Colab VM.",
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
    colab=args.colab,
    accelerator=args.accelerator,
    session_name=args.session_name,
  )


if __name__ == "__main__":
  main()
