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

import {readFileSync} from 'fs';
import {resolve} from 'path';
import * as readline from 'readline';
import {Command} from 'commander';
import {Parser, loadDefaultJapaneseParser, DEFAULT_THRES} from './parser';

/**
 * Run the command line interface program.
 * @param argv process.argv.
 */
export const cli = (argv: string[]) => {
  const program = new Command('budoux');

  program.usage('[-h] [-H] [-d STR] [-t THRES] [-m JSON] [-V] [TXT]');
  program.description(
    'BudouX is the successor to Budou, the machine learning powered line break organizer tool.'
  );
  program
    .option('-H, --html', 'HTML mode', false)
    .option('-d, --delim <str>', 'output delimiter in TEXT mode', '---')
    .option(
      '-t, --thres <number>',
      'threshold value to separate chunks',
      `${DEFAULT_THRES}`
    )
    .option('-m, --model <json>', 'custom model file path')
    .argument('[txt]', 'text');

  program.version('0.0.3');

  program.parse(argv);

  const options = program.opts();
  const {model, thres, delim, html} = options as {
    html: boolean;
    delim: string;
    thres: number;
    model?: string;
  };
  const {args} = program;

  const parser = model ? loadCustomParser(model) : loadDefaultJapaneseParser();

  switch (args.length) {
    case 0: {
      const rl = readline.createInterface({
        input: process.stdin,
      });

      let stdin = '';
      rl.on('line', line => {
        stdin += line + '\n';
      });
      process.stdin.on('end', () => {
        outputParsedTexts(parser, html, delim, thres, [stdin]);
      });
      break;
    }
    case 1: {
      outputParsedTexts(parser, html, delim, thres, args);
      break;
    }
    default: {
      throw new Error(
        'Too many arguments. Please, pass the only one argument.'
      );
    }
  }
};

/**
 * Run the command line interface program.
 * @param parser A parser.
 * @param html A flag of html output mode.
 * @param delim A delimiter to separate output sentence.
 * @param thres A threshold value to separate chunks.
 * @param args string array to parse. Array should have only one element.
 */
const outputParsedTexts = (
  parser: Parser,
  html: boolean,
  delim: string,
  thres: number,
  args: string[]
) => {
  if (html) {
    const text = args[0];
    const output = parser.translateHTMLString(text, thres);
    console.log(output);
  } else {
    const splitedTextsByNewLine = args[0]
      .split(/\r?\n/)
      .filter(text => text !== '');
    splitedTextsByNewLine.forEach((text, index) => {
      const parsedTexts = parser.parse(text, thres);
      parsedTexts.forEach(parsedText => {
        console.log(parsedText);
      });
      if (index + 1 !== splitedTextsByNewLine.length) console.log(delim);
    });
  }
};

/**
 * Loads a parser equipped with custom model.
 * @returns A parser with the loaded model.
 */
const loadCustomParser = (path: string) => {
  const file = readFileSync(resolve(path)).toString();
  const json = JSON.parse(file);
  return new Parser(new Map(Object.entries(json)));
};
