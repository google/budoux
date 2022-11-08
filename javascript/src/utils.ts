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
 * Finds the sum of the numbers in the list.
 * @param arr The list of numbers.
 * @returns The sum.
 */
export const sum = (arr: number[]) =>
  arr.reduce((prev, curr) => prev + curr, 0);

/** The separator string to specify breakpoints. */
export const SEP = '▁';

/** The invalid feature string. */
export const INVALID = '▔';
