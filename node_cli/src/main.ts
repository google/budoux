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
import yargs from 'yargs/yargs';
import {hideBin} from 'yargs/helpers';
import {loadDefaultJapaneseParser, Parser} from 'budoux';

export default async function (argv: string[]) {
  const args = await yargs(hideBin(argv))
    .usage('usage: budoux [-h] [-H] [-m JSON] [-d STR] [-V] [TXT]')
    .option('html', {
      alias: 'H',
      type: 'boolean',
      describe: 'HTML mode',
    })
    .option('delim', {
      alias: 'd',
      string: true,
      describe: 'output delimiter in TEXT mode',
    })
    .option('model', {
      alias: 'm',
      type: 'string',
      describe: 'custom model file path or json string',
      nargs: 1,
    })
    .coerce('model', (args: string) => {
      let model: Map<string, number> | undefined;

      if (args === undefined) {
        return model;
      }

      try {
        try {
          const obj = JSON.parse(args);
          model = new Map<string, number>(Object.entries(obj));
        } catch (syntaxError) {
          if (syntaxError instanceof SyntaxError) {
            const file = readFileSync(resolve(args), {
              encoding: 'utf-8',
            });
            const obj = JSON.parse(file);
            model = new Map<string, number>(Object.entries(obj));
          }
        }
      } catch (error) {
        console.error(error);
      }
      return model;
    })
    .version()
    .alias('V', 'version')
    .help()
    .parse();

  const {_, html, model, delim} = args;
  if (_.length >= 1) {
    let parser: Parser;
    if (model) {
      parser = new Parser(model);
    } else {
      parser = loadDefaultJapaneseParser();
    }

    if (html) {
      _.forEach(text => {
        const inputText = String(text);
        console.log(parser.translateHTMLString(inputText));
      });
    } else {
      _.forEach(text => {
        const inputText = String(text);
        const outputArray = parser.parse(inputText);
        outputArray.forEach((text: string, index) => {
          console.log(text);
          if (
            delim &&
            outputArray.length !== 1 &&
            index !== outputArray.length - 1
          ) {
            console.log(delim);
          }
        });
      });
    }
  } else {
    console.log('Pass one text argument to translate at least.');
    console.log('To show help: $ budoux-cli --help');
  }
}
