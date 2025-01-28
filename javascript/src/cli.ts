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
import * as path from 'path';
import * as readline from 'readline';
import {Command} from 'commander';
import {
  HTMLProcessingParser,
  loadDefaultParsers,
  loadDefaultJapaneseParser,
} from './index.js';

const CLI_VERSION = '0.7.0';
const defaultParsers = loadDefaultParsers();

/**
 * Run the command line interface program.
 * @param argv process.argv.
 */
export const cli = (argv: string[]) => {
  const program = new Command('budoux');

  program.usage('[-h] [-H] [-d STR] [-s STR] [-m JSON] [-l LANG] [-V] [TXT]');
  program.description(
    'BudouX is the successor to Budou, the machine learning powered line break organizer tool.'
  );
  program
    .option('-H, --html', 'HTML mode', false)
    .option(
      '-d, --delim <str>',
      'output sentence delimiter in TEXT mode',
      '---'
    )
    .option('-s, --sep <str>', 'output phrase separator in TEXT mode', '\n')
    .option('-m, --model <json>', 'model file path')
    .option(
      '-l, --lang <str>',
      `language model to use. -m and --model will be prioritized if any.\navailable languages: ${[
        ...defaultParsers.keys(),
      ].join(', ')}`
    )
    .argument('[txt]', 'text')
    .allowExcessArguments();

  program.version(CLI_VERSION);

  program.parse(argv);

  const options = program.opts();
  const {lang, model, delim, sep, html} = options as {
    html: boolean;
    delim: string;
    sep: string;
    model?: string;
    lang?: string;
  };
  const {args} = program;

  const parser = model
    ? loadCustomParser(model)
    : lang && defaultParsers.has(lang)
    ? defaultParsers.get(lang)!
    : loadDefaultJapaneseParser();

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
        outputParsedTexts(parser, html, delim, sep, [stdin]);
      });
      break;
    }
    case 1: {
      outputParsedTexts(parser, html, delim, sep, args);
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
 * @param sep A separator to separate output phrases.
 * @param args string array to parse. Array should have only one element.
 */
const outputParsedTexts = (
  parser: HTMLProcessingParser,
  html: boolean,
  delim: string,
  sep: string,
  args: string[]
) => {
  if (html) {
    const text = args[0];
    const output = parser.translateHTMLString(text);
    console.log(output);
  } else {
    const splitedTextsByNewLine = args[0]
      .split(/\r?\n/)
      .filter(text => text !== '');
    splitedTextsByNewLine.forEach((text, index) => {
      const parsedTexts = parser.parse(text);
      console.log(parsedTexts.join(sep));
      if (index + 1 !== splitedTextsByNewLine.length) console.log(delim);
    });
  }
};

/**
 * Loads a parser equipped with custom model.
 * @return A parser with the loaded model.
 */
const loadCustomParser = (modelPath: string) => {
  const file = readFileSync(path.resolve(modelPath)).toString();
  const model = JSON.parse(file);
  return new HTMLProcessingParser(model);
};
