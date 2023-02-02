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
import {JSDOM} from 'jsdom';
import {loadDefaultJapaneseParser} from '../parser.js';

const parser = loadDefaultJapaneseParser();

describe('Web Components', () => {
  let dom: JSDOM;

  beforeAll(async () => {
    dom = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>');
    global.customElements = dom.window.customElements;
    global.HTMLElement = dom.window.HTMLElement;
    global.MutationObserver = dom.window.MutationObserver;
    await import('../webcomponents/budoux-ja.js');
    await dom.window.customElements.whenDefined('budoux-ja');
  })

  it('should process the provided text.', () => {
    const inputText = '今日は良い天気です。'

    const budouxElement = dom.window.document.createElement('budoux-ja');
    budouxElement.textContent = inputText;
    dom.window.document.body.appendChild(budouxElement);

    const mirroredElement = dom.window.document.createElement('span');
    mirroredElement.textContent = inputText;
    parser.applyElement(mirroredElement);

    expect(budouxElement.shadowRoot!.innerHTML).toBe(mirroredElement.outerHTML);
  });

  it('should react to the text content change after attached.', (resolve) => {
    const budouxElement = dom.window.document.createElement('budoux-ja');
    budouxElement.textContent = '今日は良い天気です。';
    dom.window.document.body.appendChild(budouxElement);

    const inputText = '明日はどうなるかな。'
    const mirroredElement = dom.window.document.createElement('span');
    mirroredElement.textContent = inputText;
    parser.applyElement(mirroredElement);
    const budouxShadowRoot = budouxElement.shadowRoot!;

    const observer = new dom.window.MutationObserver(() => {
      expect(budouxShadowRoot.innerHTML).toBe(mirroredElement.outerHTML);
      resolve();
    });
    observer.observe(budouxShadowRoot, {
      childList: true,
    });

    budouxElement.textContent = inputText;
  })
});
