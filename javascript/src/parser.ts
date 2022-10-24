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

import {unicodeBlocks} from './data/unicode_blocks.js';
import {model as jaKNBCModel} from './data/models/ja-knbc.js';
import {model as zhHansModel} from './data/models/zh-hans.js';
import {parseFromString} from './dom.js';
import {HTMLProcessor} from './html_processor.js';
import {bisectRight, INVALID, sum} from './utils.js';

// We could use `Node.TEXT_NODE` and `Node.ELEMENT_NODE` in a browser context,
// but we define the same here for Node.js environments.
const NODETYPE = {
  ELEMENT: 1,
  TEXT: 3,
};

export class Parser {
  model;

  constructor(model: Map<string, number>) {
    this.model = model;
  }

  /**
   * Generates a Unicode Block feature from the given character.
   *
   * @param w A character input.
   * @returns A Unicode Block feature.
   */
  static getUnicodeBlockFeature(w: string) {
    if (!w || w === INVALID) return INVALID;
    const cp = w.codePointAt(0);
    if (cp === undefined) return INVALID;
    const bn = bisectRight(unicodeBlocks, cp);
    return `${bn}`.padStart(3, '0');
  }

  /**
   * Generates a feature from characters around (w1-w6) and past
   * results (p1-p3).
   *
   * @param w1 The character 3 characters before the break point.
   * @param w2 The character 2 characters before the break point.
   * @param w3 The character right before the break point.
   * @param w4 The character right after the break point.
   * @param w5 The character 2 characters after the break point.
   * @param w6 The character 3 characters after the break point.
   * @returns A feature to be consumed by a classifier.
   */
  static getFeature(
    w1: string,
    w2: string,
    w3: string,
    w4: string,
    w5: string,
    w6: string
  ) {
    const b1 = Parser.getUnicodeBlockFeature(w1);
    const b2 = Parser.getUnicodeBlockFeature(w2);
    const b3 = Parser.getUnicodeBlockFeature(w3);
    const b4 = Parser.getUnicodeBlockFeature(w4);
    const b5 = Parser.getUnicodeBlockFeature(w5);
    const b6 = Parser.getUnicodeBlockFeature(w6);
    const rawFeature = {
      UW1: w1,
      UW2: w2,
      UW3: w3,
      UW4: w4,
      UW5: w5,
      UW6: w6,
      BW1: w2 + w3,
      BW2: w3 + w4,
      BW3: w4 + w5,
      TW1: w1 + w2 + w3,
      TW2: w2 + w3 + w4,
      TW3: w3 + w4 + w5,
      TW4: w4 + w5 + w6,
      UB1: b1,
      UB2: b2,
      UB3: b3,
      UB4: b4,
      UB5: b5,
      UB6: b6,
      BB1: b2 + b3,
      BB2: b3 + b4,
      BB3: b4 + b5,
      TB1: b1 + b2 + b3,
      TB2: b2 + b3 + b4,
      TB3: b3 + b4 + b5,
      TB4: b4 + b5 + b6,
    };
    return Object.entries(rawFeature)
      .filter(entry => !entry[1].includes(INVALID))
      .map(([key, value]) => `${key}:${value}`);
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
    const baseScore = -sum([...this.model.values()]);

    for (let i = 1; i < sentence.length; i++) {
      const feature = Parser.getFeature(
        sentence[i - 3] || INVALID,
        sentence[i - 2] || INVALID,
        sentence[i - 1],
        sentence[i],
        sentence[i + 1] || INVALID,
        sentence[i + 2] || INVALID
      );
      const score =
        baseScore + 2 * sum(feature.map(f => this.model.get(f) || 0));
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
  return new Parser(new Map(Object.entries(jaKNBCModel)));
};

export const loadDefaultSimplifiedChineseParser = () => {
  return new Parser(new Map(Object.entries(zhHansModel)));
};
