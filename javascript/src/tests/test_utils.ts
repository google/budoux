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

import 'jasmine';
import {bisectRight} from '../utils.js';

describe('utils.bisectRight', () => {
  const arr = [1, 3, 8, 12, 34];
  const testInsertionPoint = (item: number, expectedPoint: number) => {
    const point = bisectRight(arr, item);
    expect(point).toBe(expectedPoint);
  };

  it('should find the point when the item is included.', () => {
    testInsertionPoint(8, 3);
  });

  it('should find the point when the item is not included.', () => {
    testInsertionPoint(4, 2);
  });

  it('should return zero when the item is the smallest.', () => {
    testInsertionPoint(-100, 0);
  });

  it('should return the array length when the item is the biggest', () => {
    testInsertionPoint(100, arr.length);
  });
});
