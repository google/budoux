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

import {loadDefaultJapaneseParser} from '../index.js';
import '../webcomponents/budoux-ja.js';

const parser = loadDefaultJapaneseParser();

describe('Web Components', () => {
  beforeAll(async () => {
    await window.customElements.whenDefined('budoux-ja');
  });

  it('should process the provided text.', () => {
    const inputText = '今日は良い天気です。';

    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.textContent = inputText;
    window.document.body.appendChild(budouxElement);

    const mirroredElement = window.document.createElement('span');
    mirroredElement.textContent = inputText;
    parser.applyToElement(mirroredElement);

    expect(budouxElement.innerHTML).toBe(mirroredElement.outerHTML);
  });

  it('should react to the text content change after attached.', resolve => {
    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.textContent = '今日は良い天気です。';
    window.document.body.appendChild(budouxElement);

    const inputText = '明日はどうなるかな。';
    const mirroredElement = window.document.createElement('span');
    mirroredElement.textContent = inputText;
    parser.applyToElement(mirroredElement);

    const observer = new window.MutationObserver(() => {
      expect(budouxElement.innerHTML).toBe(mirroredElement.outerHTML);
      resolve();
    });
    observer.observe(budouxElement, {
      childList: true,
    });

    budouxElement.textContent = inputText;
  });
});
