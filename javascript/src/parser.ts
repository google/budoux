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

/**
 * Base BudouX parser.
 */
export class Parser {
  /** BudouX model data */
  private readonly model: Map<string, Map<string, number>>;
  private readonly baseScore: number;

  /**
   * Constructs a BudouX parser.
   * @param model A model data.
   */
  constructor(model: {[key: string]: {[key: string]: number}}) {
    this.model = new Map(
      Object.entries(model).map(([k, v]) => [k, new Map(Object.entries(v))])
    );
    this.baseScore =
      -0.5 *
      [...this.model.values()]
        .map(group => [...group.values()])
        .flat()
        .reduce((prev, curr) => prev + curr, 0);
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
   * @param sentence An input sentence.
   * @return The list of boundaries.
   */
  parseBoundaries(sentence: string): number[] {
    const result = [];

    for (let i = 1; i < sentence.length; i++) {
      let score = this.baseScore;
      // NOTE: Score values in models may be negative.
      /* eslint-disable */
      score += this.model.get('UW1')?.get(sentence.substring(i - 3, i - 2)) || 0;
      score += this.model.get('UW2')?.get(sentence.substring(i - 2, i - 1)) || 0;
      score += this.model.get('UW3')?.get(sentence.substring(i - 1, i)) || 0;
      score += this.model.get('UW4')?.get(sentence.substring(i, i + 1)) || 0;
      score += this.model.get('UW5')?.get(sentence.substring(i + 1, i + 2)) || 0;
      score += this.model.get('UW6')?.get(sentence.substring(i + 2, i + 3)) || 0;
      score += this.model.get('BW1')?.get(sentence.substring(i - 2, i)) || 0;
      score += this.model.get('BW2')?.get(sentence.substring(i - 1, i + 1)) || 0;
      score += this.model.get('BW3')?.get(sentence.substring(i, i + 2)) || 0;
      score += this.model.get('TW1')?.get(sentence.substring(i - 3, i)) || 0;
      score += this.model.get('TW2')?.get(sentence.substring(i - 2, i + 1)) || 0;
      score += this.model.get('TW3')?.get(sentence.substring(i - 1, i + 2)) || 0;
      score += this.model.get('TW4')?.get(sentence.substring(i, i + 3)) || 0;
      /* eslint-enable */
      if (score > 0) result.push(i);
    }
    return result;
  }
}
