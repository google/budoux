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
"""Colab CLI runner module for remote accelerator training.

Implements binary resolution (env var -> PATH), remote session provisioning,
file transfer, script execution, and auto-cleanup via Python context managers.
"""

import os
import shutil
import subprocess
from typing import Any, List, Optional


def get_colab_binary() -> str:
  """Resolves colab CLI binary path.

  1. Environment variable `COLAB_CLI_PATH`
  2. System PATH lookup (`shutil.which('colab')` via PyPI `google-colab-cli`)

  Returns:
    Path to the resolved colab binary.

  Raises:
    FileNotFoundError: If no valid colab binary is found.
  """
  env_path = os.environ.get("COLAB_CLI_PATH")
  if env_path and os.path.exists(env_path):
    return env_path

  which_path = shutil.which("colab")
  if which_path:
    return which_path

  raise FileNotFoundError(
      "Colab CLI binary not found. Please install the PyPI package "
      "'google-colab-cli' (providing the 'colab' CLI executable for managing "
      "remote Colab runtimes) via: pip install google-colab-cli "
      "or set the COLAB_CLI_PATH environment variable.")


class ColabRunner:
  """Manages remote Google Colab VM execution lifecycle."""

  def __init__(
      self,
      session_name: str = "budoux-train",
      accelerator: str = "T4",
      binary_path: Optional[str] = None,
  ) -> None:
    """Initializes ColabRunner.

    Args:
      session_name: Remote session identifier.
      accelerator: Accelerator type (e.g., 'T4', 'A100', 'L4', 'v6e1', 'v5e1').
      binary_path: Optional explicit path to colab CLI binary.
    """
    self.session_name = session_name
    self.accelerator = accelerator
    self.binary_path = binary_path or get_colab_binary()
    self._is_active = False

  def _run_cmd(self,
               args: List[str],
               check: bool = True) -> subprocess.CompletedProcess[str]:
    """Executes a colab CLI command subprocess.

    Args:
      args: Command argument list.
      check: Whether to raise CalledProcessError on non-zero exit code.

    Returns:
      CompletedProcess instance.
    """
    cmd = [self.binary_path] + args
    print(f"[Colab CLI] Executing: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=False)

  def provision_session(self) -> None:
    """Provisions a new remote Colab session with specified accelerator."""
    cmd = ["new", "-s", self.session_name]
    # TPU variants start with a small 'v' (e.g. v6e1, v5e1), GPUs use --gpu flag
    if self.accelerator.startswith("v"):
      cmd.extend(["--tpu", self.accelerator])
    else:
      cmd.extend(["--gpu", self.accelerator])

    print(
        f"[Colab CLI] Provisioning remote session '{self.session_name}' "
        f"with accelerator '{self.accelerator}'...",
        flush=True,
    )
    self._run_cmd(cmd)
    self._is_active = True

  def upload_file(self,
                  local_path: str,
                  remote_path: Optional[str] = None) -> None:
    """Uploads a local file to the remote VM."""
    cmd = ["upload", "-s", self.session_name, local_path]
    if remote_path:
      cmd.append(remote_path)
    print(f"[Colab CLI] Uploading {local_path} -> remote VM...", flush=True)
    self._run_cmd(cmd)

  def download_file(self, remote_path: str, local_path: str) -> None:
    """Downloads a file from the remote VM to local workstation."""
    cmd = ["download", "-s", self.session_name, remote_path, local_path]
    print(
        f"[Colab CLI] Downloading remote {remote_path} -> {local_path}...",
        flush=True,
    )
    self._run_cmd(cmd)

  def exec_script(self,
                  script_path: str,
                  output_image: Optional[str] = None,
                  timeout: float = 43200.0) -> None:
    """Executes a local Python script remotely on the Colab VM.

    Args:
      script_path: Path to the local Python script.
      output_image: Optional local path for intercepted Matplotlib plots.
      timeout: Execution timeout in seconds (default: 43200.0).
    """
    cmd = [
        "exec",
        "-s",
        self.session_name,
        "-f",
        script_path,
        "--timeout",
        str(timeout),
    ]
    if output_image:
      cmd.extend(["--output-image", output_image])
    print(
        f"[Colab CLI] Running remote script execution ({script_path})...",
        flush=True,
    )
    self._run_cmd(cmd)


  def stop_session(self) -> None:
    """Terminates and cleans up the remote Colab session."""
    if self._is_active:
      print(
          f"[Colab CLI] Terminating remote session '{self.session_name}'...",
          flush=True,
      )
      self._run_cmd(["stop", "-s", self.session_name], check=False)
      self._is_active = False

  def __enter__(self) -> "ColabRunner":
    self.provision_session()
    return self

  def __exit__(
      self,
      exc_type: Optional[type],
      exc_val: Optional[BaseException],
      exc_tb: Optional[Any],
  ) -> None:
    self.stop_session()
