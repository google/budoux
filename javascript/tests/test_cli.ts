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

import 'jasmine';
import {promisify} from 'util';
import childProcess from 'child_process';
import {cli} from '../src/cli';
const execFile = promisify(childProcess.execFile);

describe('cli', () => {
  it('should output the wrapped HTML sentence when execute budoux command with --html option.', async () => {
    const inputText = '今日は天気です。';
    const expectedText =
      '<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>\n';
    const result = await execFile('budoux', ['--html', inputText]);
    expect(result.stdout).toBe(expectedText);
  });

  it('should output the wrapped HTML sentence when execute budoux command with -H option alias.', async () => {
    const inputText = '今日は天気です。';
    const expectedText =
      '<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>\n';
    const result = await execFile('budoux', ['-H', inputText]);
    expect(result.stdout).toBe(expectedText);
  });

  it('should output the separated sentence with separater when execute budoux command with --delim option.', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const expectedText = `今日は
天気です。
---
明日は
雨かな？
`;
    const result = await execFile('budoux', ['--delim', '---', inputText]);
    expect(result.stdout).toBe(expectedText);
  });

  it('should output the separated sentence with separater when execute budoux command with -d option alias.', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const expectedText = `今日は
天気です。
---
明日は
雨かな？
`;
    const result = await execFile('budoux', ['-d', '---', inputText]);
    expect(result.stdout).toBe(expectedText);
  });

  it('should output the separated sentence with separater when execute budoux with stdin inputed by pipe', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const expectedText = `今日は
天気です。
---
明日は
雨かな？
`;
    const result = await execFile(`echo '${inputText}' | budoux`, {
      shell: true,
    });
    expect(result.stdout).toBe(expectedText);
  });

  it('should output error message when get more than one text argument.', async () => {
    const argv = [
      'node',
      'budoux',
      '今日は天気です。',
      '明日は晴れるでしょう。',
    ];
    const stab = () => cli(argv);

    expect(stab).toThrowError(
      'Too many arguments. Please, pass the only one argument.'
    );
  });
});
