/**
 * @license
 * Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const assert = require('assert');
const path = require('path');
const childProcess = require('child_process');
const package = require('../package.json');

const packageVersion = package.version;
const runCli = args =>
  new Promise(resolve => {
    childProcess.execFile(
      'node',
      [path.resolve(__dirname, '..', 'bin', 'budoux.js'), ...args],
      (error, stdout, stderr) => {
        resolve({
          error,
          stdout,
          stderr,
        });
      }
    );
  });

runCli(['-V']).then(({stdout}) => {
  assert.equal(
    stdout.replace('\n', ''),
    packageVersion,
    'Package version and CLI version output (-V) should match.'
  );
});

runCli(['--version']).then(({stdout}) => {
  assert.equal(
    stdout.replace('\n', ''),
    packageVersion,
    'Package version and CLI version output (--version) should match.'
  );
});
