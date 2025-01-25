# Copyright 2024 Google LLC
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

import argparse
import json
import re
import subprocess


def main():
  parser = argparse.ArgumentParser(description='Bump the version number.')
  parser.add_argument(
      'new_version', type=str, help='The new version number (e.g., 1.2.3)')
  args = parser.parse_args()
  new_version = args.new_version

  # Updates Python port version number
  init_file = 'budoux/__init__.py'
  with open(init_file, 'r') as f:
    content = f.read()
  new_content = re.sub(r'(__version__\s+=\s+[\'"])([\.\d]+)([\'"])',
                       rf'\g<1>{new_version}\g<3>', content)
  with open(init_file, 'w') as f:
    f.write(new_content)

  # Updates JavaScript port version number
  package_json_path = 'javascript/package.json'
  with open(package_json_path, 'r') as f:
    package_data = json.load(f)
    current_version = package_data.get('version')

  if current_version != new_version:
    npm_command = ['npm', 'version', new_version, '--no-git-tag-version']
    subprocess.run(npm_command, cwd='javascript', check=True)
  else:
    print(f"JavaScript version is already {new_version}, skipping npm version.")

  cli_file = 'javascript/src/cli.ts'
  with open(cli_file, 'r') as f:
    content = f.read()
  new_content = re.sub(r'(const\s+CLI_VERSION\s+=\s+[\'"])([\.\d]+)([\'"])',
                       rf'\g<1>{new_version}\g<3>', content)
  with open(cli_file, 'w') as f:
    f.write(new_content)

  # Updates Java port version number
  mvn_command = [
      'mvn', 'versions:set', f'-DnewVersion={new_version}',
      '-DgenerateBackupPoms=false'
  ]
  subprocess.run(mvn_command, cwd='java', check=True)


if __name__ == "__main__":
  main()
