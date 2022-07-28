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

type execFileCallBack = {
  error: ExecFileException | null;
  stdout: string;
  stderr: string;
};

const runCli = (args: string[]): Promise<execFileCallBack> => {
  return new Promise(resolve => {
    const binPath = path.resolve('./bin/budoux.js');
    execFile('node', [binPath, ...args], (error, stdout, stderr) => {
      resolve({
        error,
        stdout,
        stderr,
      });
    });
  });
};

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
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
    const argv = ['node', 'budoux', '--model', customModelPath, inputText];
    const expectedStdOuts = 'abcde\nabcd'.split('\n');
    cli(argv);
    expectedStdOuts.forEach(stdout => {
      expect(console.log).toHaveBeenCalledWith(stdout);
    });
  });

  it('should output the separated sentence with custom model when execute budoux command with -m option alias.', () => {
    const inputText = 'abcdeabcd';
    const customModelPath = path.resolve(
      __dirname,
      'models',
      'separate_right_before_a.json'
    );
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
    const runCliWithStdin = (stdin: string): Promise<execFileCallBack> => {
      return new Promise(resolve => {
        const binPath = path.resolve('./bin/budoux.js');
        const child = execFile('node', [binPath], (error, stdout, stderr) => {
          resolve({
            error,
            stdout,
            stderr,
          });
        });
        const stdinStream = new stream.Readable();
        stdinStream.push(stdin);
        stdinStream.push(null);
        if (child.stdin) {
          stdinStream.pipe(child.stdin);
        }
      });
    };

    const {stdout} = await runCliWithStdin('今日は天気です。\n明日は雨かな？');
    const expectedStdOut = '今日は\n天気です。\n---\n明日は\n雨かな？\n';
    expect(stdout).toBe(expectedStdOut);
  }, 3000);

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
  }, 3000);
});
