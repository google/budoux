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
import {execSync} from 'child_process';
describe('main', () => {
  it('should output the separated sentence when exec command with no option.', () => {
    const inputText = '今日は天気です。';
    const expectedText = `今日は
天気です。
`;
    const outputText = String(execSync(`npx budoux-cli ${inputText}`));
    expect(outputText).toBe(expectedText);
  });

  it('should output the message when exec command with no argunent.', () => {
    const expectedText = `Pass one text argument to translate at least.
To show help: $ budoux-cli --help
`;
    const outputText = String(execSync('npx budoux-cli'));
    expect(outputText).toBe(expectedText);
  });

  it('should output the wrapped HTML sentence when exec command with --html option.', () => {
    const inputText = '今日は天気です。';
    const expectedText = `<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>
`;
    const outputText = String(execSync(`npx budoux-cli --html ${inputText}`));
    expect(outputText).toBe(expectedText);
  });

  it('should output the wrapped HTML sentence when exec command with -H option alias.', () => {
    const inputText = '今日は天気です。';
    const expectedText = `<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>
`;
    const outputText = String(execSync(`npx budoux-cli -H ${inputText}`));
    expect(outputText).toBe(expectedText);
  });

  it('should output the separated sentence with separater when exec command with --delim option.', () => {
    const inputText = '今日は天気です。';
    const expectedText = `今日は
---
天気です。
`;
    const outputText = String(
      execSync(`npx budoux-cli --delim '---' ${inputText}`)
    );
    expect(outputText).toBe(expectedText);
  });

  it('should output the separated sentence with separater when exec command with -d option alias.', () => {
    const inputText = '今日は天気です。';
    const expectedText = `今日は
---
天気です。
`;
    const outputText = String(execSync(`npx budoux-cli -d '---' ${inputText}`));
    expect(outputText).toBe(expectedText);
  });

  it('should output the version when exec command with --version option.', () => {
    const expectedText = `0.0.1
`;
    const outputText = String(execSync('npx budoux-cli --version'));
    expect(outputText).toBe(expectedText);
  });

  it('should output the version when exec command with -V option.', () => {
    const expectedText = `0.0.1
`;
    const outputText = String(execSync('npx budoux-cli -V'));
    expect(outputText).toBe(expectedText);
  });

  it('should output the help when exec command with --help option.', () => {
    const expectedText = `usage: budoux [-h] [-H] [-m JSON] [-d STR] [-V] [TXT]

オプション:
  -H, --html     HTML mode                                                [真偽]
  -d, --delim    output delimiter in TEXT mode                          [文字列]
  -m, --model    custom model file path or json string                  [文字列]
  -V, --version  バージョンを表示                                         [真偽]
  -h, --help     ヘルプを表示                                             [真偽]
`;
    const outputText = String(execSync('npx budoux-cli --help'));
    expect(outputText).toBe(expectedText);
  });

  it('should output the help when exec command with -h option alias.', () => {
    const expectedText = `usage: budoux [-h] [-H] [-m JSON] [-d STR] [-V] [TXT]

オプション:
  -H, --html     HTML mode                                                [真偽]
  -d, --delim    output delimiter in TEXT mode                          [文字列]
  -m, --model    custom model file path or json string                  [文字列]
  -V, --version  バージョンを表示                                         [真偽]
  -h, --help     ヘルプを表示                                             [真偽]
`;
    const outputText = String(execSync('npx budoux-cli --help'));
    expect(outputText).toBe(expectedText);
  });
});
