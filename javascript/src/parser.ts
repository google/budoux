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
import {parseFromString} from './dom.js';
import {HTMLProcessor} from './html_processor.js';
import {sum} from './utils.js';

// We could use `Node.TEXT_NODE` and `Node.ELEMENT_NODE` in a browser context,
// but we define the same here for Node.js environments.
const NODETYPE = {
  ELEMENT: 1,
  TEXT: 3,
};

export class Parser {
  model;

  constructor(model: {[key: string]: {[key: string]: number}}) {
    this.model = new Map(
      Object.entries(model).map(([k, v]) => [k, new Map(Object.entries(v))])
    );
  }

  /**
   * Checks if the given element has a text node in its children.
   *
   * @param ele An element to be checked.
   * @returns Whether the element has a child text node.
   */
  static hasChildTextNode(ele: HTMLElement) {
    for (const child of ele.childNodes) {
      if (child.nodeType === NODETYPE.TEXT) return true;
    }
    return false;
  }

  /**
   * Parses the input sentence and returns a list of semantic chunks.
   *
   * @param sentence An input sentence.
   * @returns The retrieved chunks.
   */
  parse(sentence: string) {
    if (sentence === '') return [];
    const result = [sentence[0]];
    const baseScore =
      -0.5 *
      sum([...this.model.values()].map(group => [...group.values()]).flat());

    for (let i = 1; i < sentence.length; i++) {
      let score = baseScore;
      score += this.model.get('UW1')?.get(sentence.slice(i - 3, i - 2)) || 0;
      score += this.model.get('UW2')?.get(sentence.slice(i - 2, i - 1)) || 0;
      score += this.model.get('UW3')?.get(sentence.slice(i - 1, i)) || 0;
      score += this.model.get('UW4')?.get(sentence.slice(i, i + 1)) || 0;
      score += this.model.get('UW5')?.get(sentence.slice(i + 1, i + 2)) || 0;
      score += this.model.get('UW6')?.get(sentence.slice(i + 2, i + 3)) || 0;
      score += this.model.get('BW1')?.get(sentence.slice(i - 2, i)) || 0;
      score += this.model.get('BW2')?.get(sentence.slice(i - 1, i + 1)) || 0;
      score += this.model.get('BW3')?.get(sentence.slice(i, i + 2)) || 0;
      score += this.model.get('TW1')?.get(sentence.slice(i - 3, i)) || 0;
      score += this.model.get('TW2')?.get(sentence.slice(i - 2, i + 1)) || 0;
      score += this.model.get('TW3')?.get(sentence.slice(i - 1, i + 2)) || 0;
      score += this.model.get('TW4')?.get(sentence.slice(i, i + 3)) || 0;
      if (score > 0) result.push('');
      result[result.length - 1] += sentence[i];
    }
    return result;
  }

  /**
   * Applies markups for semantic line breaks to the given HTML element.
   * @param parentElement The input element.
   */
  applyElement(parentElement: HTMLElement) {
    const htmlProcessor = new HTMLProcessor(this, {
      separator: parentElement.ownerDocument.createElement('wbr'),
    });
    htmlProcessor.applyToElement(parentElement);
  }

  /**
   * Translates the given HTML string to another HTML string with markups
   * for semantic line breaks.
   * @param html An input html string.
   * @returns The translated HTML string.
   */
  translateHTMLString(html: string) {
    if (html === '') return html;
    const doc = parseFromString(html);
    if (Parser.hasChildTextNode(doc.body)) {
      const wrapper = doc.createElement('span');
      wrapper.append(...doc.body.childNodes);
      doc.body.append(wrapper);
    }
    this.applyElement(doc.body.childNodes[0] as HTMLElement);
    return doc.body.innerHTML;
  }
}

/**
 * Loads a parser equipped with the default Japanese model.
 * @returns A parser with the default Japanese model.
 */
export const loadDefaultJapaneseParser = () => {
  return new Parser(jaModel);
};

/**
 * Loads a parser equipped with the default Simplified Chinese model.
 * @returns A parser with the default Simplified Chinese model.
 */
export const loadDefaultSimplifiedChineseParser = () => {
  return new Parser(zhHansModel);
};

/**
 * Loads a parser equipped with the default Traditional Chinese model.
 * @returns A parser with the default Traditional Chinese model.
 */
export const loadDefaultTraditionalChineseParser = () => {
  return new Parser(zhHantModel);
};

/**
 * Loads available default parsers.
 * @returns A map between available lang codes and their default parsers.
 */
export const loadDefaultParsers = () => {
  return new Map([
    ['ja', loadDefaultJapaneseParser()],
    ['zh-hans', loadDefaultSimplifiedChineseParser()],
    ['zh-hant', loadDefaultTraditionalChineseParser()],
  ]);
};
