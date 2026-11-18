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

import {htmlSpec} from './data/html_spec.js';

const INVISIBLE_FORMAT_CHARS = new Set(htmlSpec.invisible_format_chars);

/**
 * Base BudouX parser.
 */
export class Parser {
  /** BudouX model data */
  private readonly model: Map<string, Map<string, number>>;
  private readonly totalScore: number;

  /**
   * Constructs a BudouX parser.
   * @param model A model data.
   */
  constructor(model: {[key: string]: {[key: string]: number}}) {
    this.model = new Map(
      Object.entries(model).map(([k, v]: [string, {[key: string]: number}]) => [
        k,
        new Map(Object.entries(v)),
      ])
    );
    this.totalScore = [...this.model.values()]
      .flatMap((group: Map<string, number>) => [...group.values()])
      .reduce((prev: number, curr: number) => prev + curr, 0);
  }

  /**
   * Parses the input sentence and returns a list of semantic chunks.
   *
   * @param sentence An input sentence.
   * @return The retrieved chunks.
   */
  parse(sentence: string): string[] {
    if (sentence === '') return [];
    const boundaries = this.parseBoundaries(sentence);
    const result = [];
    let start = 0;
    for (const boundary of boundaries) {
      result.push(sentence.slice(start, boundary));
      start = boundary;
    }
    result.push(sentence.slice(start));
    return result;
  }

  /**
   * Parses the input sentence and returns a list of boundaries.
   *
   * Invisible format characters (such as U+200B ZERO WIDTH SPACE) are treated
   * as transparent in the feature window so repeated segmentation stays
   * idempotent.
   *
   * @param sentence An input sentence.
   * @return The list of UTF-16 code unit offsets of the boundaries.
   */
  parseBoundaries(sentence: string): number[] {
    const result: number[] = [];
    if (sentence === '') return result;

    const uw1 = this.model.get('UW1');
    const uw2 = this.model.get('UW2');
    const uw3 = this.model.get('UW3');
    const uw4 = this.model.get('UW4');
    const uw5 = this.model.get('UW5');
    const uw6 = this.model.get('UW6');
    const bw1 = this.model.get('BW1');
    const bw2 = this.model.get('BW2');
    const bw3 = this.model.get('BW3');
    const tw1 = this.model.get('TW1');
    const tw2 = this.model.get('TW2');
    const tw3 = this.model.get('TW3');
    const tw4 = this.model.get('TW4');

    const chars: string[] = [];
    const offsets: number[] = [];
    let utf16Offset = 0;
    for (const ch of sentence) {
      if (INVISIBLE_FORMAT_CHARS.has(ch)) {
        utf16Offset += ch.length;
        continue;
      }
      chars.push(ch);
      offsets.push(utf16Offset);
      utf16Offset += ch.length;
    }

    for (let i = 1; i < chars.length; i++) {
      let score = -this.totalScore;
      // NOTE: Score values in models may be negative.
      if (i > 2) {
        score += 2 * (uw1?.get(chars[i - 3]) ?? 0);
        score +=
          2 * (tw1?.get(chars[i - 3] + chars[i - 2] + chars[i - 1]) ?? 0);
      }
      if (i > 1) {
        score += 2 * (uw2?.get(chars[i - 2]) ?? 0);
        score += 2 * (bw1?.get(chars[i - 2] + chars[i - 1]) ?? 0);
        score += 2 * (tw2?.get(chars[i - 2] + chars[i - 1] + chars[i]) ?? 0);
      }
      score += 2 * (uw3?.get(chars[i - 1]) ?? 0);
      score += 2 * (uw4?.get(chars[i]) ?? 0);
      score += 2 * (bw2?.get(chars[i - 1] + chars[i]) ?? 0);
      if (i + 1 < chars.length) {
        score += 2 * (uw5?.get(chars[i + 1]) ?? 0);
        score += 2 * (bw3?.get(chars[i] + chars[i + 1]) ?? 0);
        score += 2 * (tw3?.get(chars[i - 1] + chars[i] + chars[i + 1]) ?? 0);
      }
      if (i + 2 < chars.length) {
        score += 2 * (uw6?.get(chars[i + 2]) ?? 0);
        score += 2 * (tw4?.get(chars[i] + chars[i + 1] + chars[i + 2]) ?? 0);
      }
      if (score > 0) result.push(offsets[i]);
    }
    return result;
  }
}
