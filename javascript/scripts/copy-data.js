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

const path = require('path');
const fs = require('fs');

const PROJECT_ROOT = path.join(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_ROOT, 'javascript', 'src', 'data');
fs.mkdirSync(path.join(DATA_DIR, 'models'), {recursive: true});

const copyUnicodeBlocks = () => {
  const sourcePath = path.join(PROJECT_ROOT, 'budoux', 'unicode_blocks.json');
  const targetPath = path.join(DATA_DIR, 'unicode_blocks.ts');
  fs.writeFileSync(
    targetPath,
    `export const unicodeBlocks = ${fs.readFileSync(sourcePath)}`
  );
};

const copyModels = () => {
  const modelsDirPath = path.join(PROJECT_ROOT, 'budoux', 'models');
  const files = fs.readdirSync(modelsDirPath);
  files.forEach(file => {
    const ext = file.split('.').pop();
    const body = file.split('.').slice(0, -1).join('.');
    if (ext !== 'json') return;
    const sourcePath = path.join(modelsDirPath, file);
    const targetPath = path.join(DATA_DIR, 'models', `${body}.ts`);
    const content = fs.readFileSync(sourcePath);
    fs.writeFileSync(
      targetPath,
      `export const model: {[key:string]: number} = ${content}`
    );
  });
};

const copySkipNodes = () => {
  const sourcePath = path.join(PROJECT_ROOT, 'budoux', 'skip_nodes.json');
  const targetPath = path.join(DATA_DIR, 'skip_nodes.ts');
  fs.writeFileSync(
    targetPath,
    `export const skipNodes = ${fs.readFileSync(sourcePath)}`
  );
};

const main = () => {
  copyUnicodeBlocks();
  copyModels();
  copySkipNodes();
};

main();
