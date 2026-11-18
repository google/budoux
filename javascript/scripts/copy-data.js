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

const copyModels = () => {
  const modelsDirPath = path.join(PROJECT_ROOT, 'budoux', 'models');
  const files = fs.readdirSync(modelsDirPath);
  files.forEach(file => {
    const ext = file.split('.').pop();
    const body = file.split('.').slice(0, -1).join('.');
    if (ext !== 'json') return;
    const sourcePath = path.join(modelsDirPath, file);
    const targetPath = path.join(DATA_DIR, 'models', `${body}.ts`);
    const content = fs.readFileSync(sourcePath, 'utf-8');
    fs.writeFileSync(
      targetPath,
      `export const model: {[key:string]: {[key:string]: number}} = ${content}`
    );
  });
};

const copyHtmlSpec = () => {
  const sourcePath = path.join(PROJECT_ROOT, 'budoux', 'html_spec.json');
  const targetPath = path.join(DATA_DIR, 'html_spec.ts');
  const content = fs.readFileSync(sourcePath, 'utf-8').trimEnd();
  const tsContent = `export interface HtmlSpec {
  element_actions: {
    block: string[];
    skip: string[];
    nobreak: string[];
    break: string[];
    break_opportunity: string[];
  };
  void_elements: string[];
  invisible_format_chars: string[];
  space_chars: string[];
}

export const htmlSpec: HtmlSpec = ${content};
`;
  fs.writeFileSync(targetPath, tsContent);
};

const main = () => {
  copyModels();
  copyHtmlSpec();
};

main();
