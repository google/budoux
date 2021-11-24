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

import yargs from 'yargs/yargs';
import {hideBin} from 'yargs/helpers';
import {loadDefaultJapaneseParser} from 'budoux';

export default async function (argv: string[]) {
  const args = await yargs(hideBin(argv))
    .usage('usage: budoux [-h] [-H] [-m JSON] [-d STR] [-V] [TXT]')
    .option('html', {
      alias: 'H',
      type: 'boolean',
      describe: 'HTML mode',
    })
    .version()
    .help()
    .parse();

  const {_, html} = args;
  if (_.length >= 1) {
    const parser = loadDefaultJapaneseParser();

    if (html) {
      _.forEach(text => {
        const inputText = String(text);
        console.log(parser.translateHTMLString(inputText));
      });
    } else {
      _.forEach(text => {
        const inputText = String(text);
        const outputArray = parser.parse(inputText);
        outputArray.forEach((text: string) => {
          console.log(text);
        });
      });
    }
  } else {
    console.log('Please, pass one text argument to translate at least.');
  }
}
