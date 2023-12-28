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

import {model as jaModel} from './data/models/ja.js';
import {model as zhHansModel} from './data/models/zh-hans.js';
import {model as zhHantModel} from './data/models/zh-hant.js';
import {model as thModel} from './data/models/th.js';
import {HTMLProcessingParser} from './html_processor.js';

export {Parser} from './parser.js';
export {HTMLProcessor, HTMLProcessingParser} from './html_processor.js';
export {jaModel, zhHansModel, zhHantModel};

/**
 * Loads a parser equipped with the default Japanese model.
 * @return A parser with the default Japanese model.
 */
export const loadDefaultJapaneseParser = () => {
  return new HTMLProcessingParser(jaModel);
};

/**
 * Loads a parser equipped with the default Simplified Chinese model.
 * @return A parser with the default Simplified Chinese model.
 */
export const loadDefaultSimplifiedChineseParser = () => {
  return new HTMLProcessingParser(zhHansModel);
};

/**
 * Loads a parser equipped with the default Traditional Chinese model.
 * @return A parser with the default Traditional Chinese model.
 */
export const loadDefaultTraditionalChineseParser = () => {
  return new HTMLProcessingParser(zhHantModel);
};

/**
 * Loads a parser equipped with the default Thai model.
 * @returns A parser with the default Thai model.
 */
export const loadDefaultThaiParser = () => {
  return new HTMLProcessingParser(thModel);
};
/**
 * Loads available default parsers.
 * @return A map between available lang codes and their default parsers.
 */
export const loadDefaultParsers = () => {
  return new Map([
    ['ja', loadDefaultJapaneseParser()],
    ['zh-hans', loadDefaultSimplifiedChineseParser()],
    ['zh-hant', loadDefaultTraditionalChineseParser()],
    ['th', loadDefaultThaiParser()],
  ]);
};
