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
import childProcess, {spawn} from 'child_process';
import {cli} from '../src/cli';
import {version} from '../package.json';
const execFile = promisify(childProcess.execFile);

describe('cli', () => {
  beforeEach(() => {
    spyOn(console, 'log');
  });

  it('should output the wrapped HTML sentence when execute budoux command with --html option.', () => {
    const inputText = '今日は天気です。';
    const argv = ['node', 'budoux', '--html', inputText];
    const expectedStdOut =
      '<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>';
    cli(argv);
    expect(console.log).toHaveBeenCalledWith(expectedStdOut);
  });

  it('should output the wrapped HTML sentence when execute budoux command with -H option alias.', () => {
    const inputText = '今日は天気です。';
    const argv = ['node', 'budoux', '-H', inputText];
    const expectedStdOut =
      '<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<wbr>天気です。</span>';
    cli(argv);
    expect(console.log).toHaveBeenCalledWith(expectedStdOut);
  });

  it('should output the separated sentence with custom model when execute budoux command with --model option.', () => {
    const inputText = 'abcdeabcd';
    const customModelPath = 'tests/models/separate_right_before_a.json';
    const argv = ['node', 'budoux', '--model', customModelPath, inputText];
    const expectedStdOuts = 'abcde\nabcd'.split('\n');
    cli(argv);
    expectedStdOuts.forEach(stdout => {
      expect(console.log).toHaveBeenCalledWith(stdout);
    });
  });

  it('should output the separated sentence with custom model when execute budoux command with -m option alias.', () => {
    const inputText = 'abcdeabcd';
    const customModelPath = 'tests/models/separate_right_before_a.json';
    const argv = ['node', 'budoux', '-m', customModelPath, inputText];
    const expectedStdOuts = 'abcde\nabcd'.split('\n');
    cli(argv);
    expectedStdOuts.forEach(stdout => {
      expect(console.log).toHaveBeenCalledWith(stdout);
    });
  });

  it('should output the separated sentence with separater when execute budoux command with --delim option.', () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const argv = ['node', 'budoux', '--delim', '---', inputText];
    const expectedStdOuts = '今日は\n天気です。\n---\n明日は\n雨かな？'.split(
      '\n'
    );
    cli(argv);
    expectedStdOuts.forEach(stdout => {
      expect(console.log).toHaveBeenCalledWith(stdout);
    });
  });

  it('should output the separated sentence with separater when execute budoux command with -d option alias.', () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const argv = ['node', 'budoux', '--delim', '---', inputText];
    const expectedStdOuts = '今日は\n天気です。\n---\n明日は\n雨かな？'.split(
      '\n'
    );
    cli(argv);
    expectedStdOuts.forEach(stdout => {
      expect(console.log).toHaveBeenCalledWith(stdout);
    });
  });

  it('should output the separated sentence with separater when execute budoux with stdin inputed by pipe', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const expectedText = '今日は\n天気です。\n---\n明日は\n雨かな？\n';
    const result = await execFile(`echo '${inputText}' | budoux`, {
      shell: true,
    });
    expect(result.stdout).toBe(expectedText);
  });

  it('should output the error message when get more than one text argument.', async () => {
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

  it('should output the error message when get extra option argument.', async () => {
    const argv = [
      'node',
      'budoux',
      '--delim',
      '---',
      '<extra delimiter option arguments>',
      '今日は天気です。',
    ];
    const stab = () => cli(argv);

    expect(stab).toThrowError(
      'Too many arguments. Please, pass the only one argument.'
    );
  });

  it('should output the error message when get extra option argument.', async () => {
    const argv = [
      'node',
      'budoux',
      '--model',
      'tests/models/separate_right_before_a.json',
      '<extra model option arguments>',
      '今日は天気です。',
    ];
    const stab = () => cli(argv);

    expect(stab).toThrowError(
      'Too many arguments. Please, pass the only one argument.'
    );
  });

  it('should output the version number when execute budoux command with --veriosn option.', async () => {
    const result = await execFile('budoux', ['--version']);
    expect(result.stdout).toBe(`${version}\n`);
  });

  it('should output the version number when execute budoux command with -V option alias.', async () => {
    const result = await execFile('budoux', ['-V']);
    expect(result.stdout).toBe(`${version}\n`);
  });

  it('should output the unknown option error when execute budoux command with -v option.', async () => {
    const execution = spawn('budoux', ['-v']);

    execution.stderr.on('data', stream => {
      const stderr = stream.toString();
      expect(stderr).toBe("error: unknown option '-v'\n");
      execution.kill();
    });
  });
});
