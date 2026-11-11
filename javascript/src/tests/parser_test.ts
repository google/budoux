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

import {describe, expect, it} from 'vitest';
import {thModel} from '../index.js';
import {Parser} from '../parser.js';

describe('Parser.parse', () => {
  const TEST_SENTENCE = 'abcdeabcd';

  it('should separate if a strong feature item supports.', () => {
    const model = {
      UW4: {a: 10000}, // means "should separate right before 'a'".
    };
    const parser = new Parser(model);
    const result = parser.parse(TEST_SENTENCE);
    expect(result).toEqual(['abcde', 'abcd']);
  });

  it('should separate even if it makes a phrase of one character.', () => {
    const model = {
      UW4: {b: 10000}, // means "should separate right before 'b'".
    };
    const parser = new Parser(model);
    const result = parser.parse(TEST_SENTENCE);
    expect(result).toEqual(['a', 'bcdea', 'bcd']);
  });

  it('should not separate inside a surrogate pair.', () => {
    const model = {
      UW4: {a: -10000}, // means "should separate everywhere except before 'a'".
    };
    const parser = new Parser(model);
    const result = parser.parse('𠮷野家');
    expect(result).toEqual(['𠮷', '野', '家']);
  });

  it('should match features with surrogate pairs.', () => {
    const model = {
      UW3: {𠮷: 1000},
      BW2: {𠮷野: 1000},
    };
    const parser = new Parser(model);
    const result = parser.parse('𠮷野家');
    expect(result).toEqual(['𠮷', '野家']);
  });

  it('should track cumulative offsets with multiple surrogate pairs.', () => {
    const modelBreakAfterSecond = {
      BW2: {𠮷家: 1000},
    };
    const parser1 = new Parser(modelBreakAfterSecond);
    expect(parser1.parseBoundaries('𠮷野𠮷家')).toEqual([5]);
    expect(parser1.parse('𠮷野𠮷家')).toEqual(['𠮷野𠮷', '家']);

    const modelBreakAfterBoth = {
      UW3: {𠮷: 1000},
    };
    const parser2 = new Parser(modelBreakAfterBoth);
    expect(parser2.parseBoundaries('𠮷野𠮷家')).toEqual([2, 5]);
    expect(parser2.parse('𠮷野𠮷家')).toEqual(['𠮷', '野𠮷', '家']);
  });

  it('should match outer window features across an intervening surrogate pair.', () => {
    const modelUW2 = {
      UW2: {あ: 1000},
    };
    const parserUW2 = new Parser(modelUW2);
    expect(parserUW2.parse('あ𠮷い')).toEqual(['あ𠮷', 'い']);

    const modelUW5 = {
      UW5: {い: 1000},
    };
    const parserUW5 = new Parser(modelUW5);
    expect(parserUW5.parse('あ𠮷い')).toEqual(['あ', '𠮷い']);
  });

  it('should return an empty list when the input is a blank string.', () => {
    const parser = new Parser({});
    const result = parser.parse('');
    expect(result).toEqual([]);
  });
});

describe('thModel', () => {
  it('should be exported to create a Thai parser.', () => {
    const parser = new Parser(thModel);
    const result = parser.parse('วันนี้อากาศดี');
    expect(result).toEqual(['วัน', 'นี้', 'อากาศ', 'ดี']);
  });
});
