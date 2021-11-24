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
 * Finds the insertion point maintaining the sorted order with a basic
 * bisection algorithm. This works the same as Python's bisect.bisect_right
 * method.
 *
 * @param arr The sorted array.
 * @param i The item to check the insertion point.
 * @returns The insertion point.
 */
export const bisectRight = (arr: number[], i: number): number => {
  const mid = Math.floor(arr.length / 2);
  if (i === arr[mid]) {
    return mid + 1;
  } else if (i < arr[mid]) {
    if (arr.length === 1) return 0;
    return bisectRight(arr.slice(0, mid), i);
  } else {
    if (arr.length === 1) return 1;
    return mid + bisectRight(arr.slice(mid), i);
  }
};

/** The separator string to specify breakpoints. */
export const SEP = '▁';
