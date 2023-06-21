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

  it('should return an empty list when the input is a blank string.', () => {
    const parser = new Parser({});
    const result = parser.parse('');
    expect(result).toEqual([]);
  });
});
