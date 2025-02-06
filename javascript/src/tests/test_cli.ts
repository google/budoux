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

import {cli} from '../cli.js';
import {execFile, ExecFileException} from 'child_process';
import * as path from 'path';
import stream from 'stream';
import {loadDefaultParsers} from '../index.js';

type execFileCallBack = {
  error: ExecFileException | null;
  stdout: string;
  stderr: string;
};

const runCli = (args: string[], stdin?: string): Promise<execFileCallBack> => {
  return new Promise(resolve => {
    const binPath = path.resolve('./bin/budoux.js');
    const child = execFile(
      'node',
      [binPath, ...args],
      (error, stdout, stderr) => {
        resolve({
          error,
          stdout,
          stderr,
        });
      }
    );

    if (stdin) {
      const stdinStream = new stream.Readable();
      stdinStream.push(stdin);
      stdinStream.push(null);
      if (child.stdin) {
        stdinStream.pipe(child.stdin);
      }
    }
  });
};

describe('cli', () => {
  it('should output the wrapped HTML sentence when execute budoux command with --html option.', async () => {
    const inputText = '今日は天気です。';
    const argv = ['--html', inputText];
    const expectedStdOut =
      '<span style="word-break:keep-all;overflow-wrap:anywhere">今日は\u200B天気です。</span>';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the wrapped HTML sentence when execute budoux command with -H option alias.', async () => {
    const inputText = '今日は天気です。';
    const argv = ['-H', inputText];
    const expectedStdOut =
      '<span style="word-break:keep-all;overflow-wrap:anywhere">今日は\u200B天気です。</span>';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the separated sentence with custom model when execute budoux command with --model option.', async () => {
    const inputText = 'abcdeabcd';
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
    const argv = ['--model', customModelPath, inputText];
    const expectedStdOut = 'abcde\nabcd';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the separated sentence with custom model when execute budoux command with -m option alias.', async () => {
    const inputText = 'abcdeabcd';
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
    const argv = ['-m', customModelPath, inputText];
    const expectedStdOut = 'abcde\nabcd';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should use the corresponding language model when the -l parameter is given.', async () => {
    const inputTextHans = '我们的使命是整合全球信息，供大众使用，让人人受益。';
    const expectedStdOut = loadDefaultParsers()
      .get('zh-hans')!
      .parse(inputTextHans)
      .join('\n');
    const argv = ['-l', 'zh-hans', inputTextHans];
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should use the corresponding language model when the --lang parameter is given.', async () => {
    const inputTextHans = '我們的使命是匯整全球資訊，供大眾使用，使人人受惠。';
    const expectedStdOut = loadDefaultParsers()
      .get('zh-hant')!
      .parse(inputTextHans)
      .join('\n');
    const argv = ['--lang', 'zh-hant', inputTextHans];
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should prioritize -m and --model over -l and --lang', async () => {
    const inputTextHans = '我們的使a命';
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
    const argv = [
      '--model',
      customModelPath,
      '--lang',
      'zh-hant',
      inputTextHans,
    ];
    const expectedStdOut = '我們的使\na命';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the separated sentence with separater when execute budoux command with --delim option.', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const argv = ['--delim', '###', inputText];
    const expectedStdOut = '今日は\n天気です。\n###\n明日は\n雨かな？';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the separated sentence with separater when execute budoux command with -d option alias.', async () => {
    const inputText = '今日は天気です。\n明日は雨かな？';
    const argv = ['-d', '###', inputText];
    const expectedStdOut = '今日は\n天気です。\n###\n明日は\n雨かな？';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the separated sentence with separater when execute budoux with stdin inputed by pipe', async () => {
    const {stdout} = await runCli([], '今日は天気です。\n明日は雨かな？');
    const expectedStdOut = '今日は\n天気です。\n---\n明日は\n雨かな？';
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output phrases with the separator specified by -s option', async () => {
    const inputText = '今日は天気です。';
    const argv = ['-s', '/', inputText];
    const expectedStdOut = '今日は/天気です。';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output phrases with the separator specified by --sep option', async () => {
    const inputText = '今日は天気です。';
    const argv = ['--sep', '/', inputText];
    const expectedStdOut = '今日は/天気です。';
    const {stdout} = await runCli(argv);
    expect(stdout.trim()).toBe(expectedStdOut);
  });

  it('should output the error message when get more than one text argument.', () => {
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

  it('should output the error message when get extra option argument.', () => {
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

  it('should output the error message when get extra option argument.', () => {
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
    const argv = [
      'node',
      'budoux',
      '--model',
      customModelPath,
      '<extra model option arguments>',
      '今日は天気です。',
    ];
    const stab = () => cli(argv);

    expect(stab).toThrowError(
      'Too many arguments. Please, pass the only one argument.'
    );
  });

  it('should output the unknown option error when execute budoux command with -v option.', async () => {
    const {stderr} = await runCli(['-v']);

    expect(stderr).toBe("error: unknown option '-v'\n");
  });
});
