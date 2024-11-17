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

import '../webcomponents/budoux-ja.js';

describe('Web Components', () => {
  beforeAll(async () => {
    await window.customElements.whenDefined('budoux-ja');
  });

  beforeEach(() => {
    window.document.body.innerText = '';
  });

  it('should process the provided text.', () => {
    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.textContent = '今日は良い天気です。';
    window.document.body.appendChild(budouxElement);

    expect(budouxElement.innerHTML).toBe('今日は\u200B良い\u200B天気です。');
  });

  it('should react to text content changes after attached.', resolve => {
    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.textContent = '今日は良い天気です。';
    window.document.body.appendChild(budouxElement);

    const observer = new window.MutationObserver(() => {
      expect(budouxElement.innerHTML).toBe('明日は\u200B晴れるかな？');
      resolve();
    });
    observer.observe(budouxElement, {
      childList: true,
    });
    budouxElement.textContent = '明日は晴れるかな？';
  });

  it('should work with HTML inputs.', () => {
    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.appendChild(window.document.createTextNode('昨日は'));
    const b = window.document.createElement('b');
    b.textContent = '雨';
    budouxElement.appendChild(b);
    budouxElement.appendChild(window.document.createTextNode('でした。'));
    window.document.body.appendChild(budouxElement);
    expect(budouxElement.innerHTML).toBe('昨日は<b>\u200B雨</b>でした。');
  });

  it('should have wrapping styles to control line breaks.', () => {
    const budouxElement = window.document.createElement('budoux-ja');
    budouxElement.textContent = 'Hello world';
    window.document.body.appendChild(budouxElement);
    const styles = budouxElement.computedStyleMap();
    expect(styles.get('word-break')?.toString()).toBe('keep-all');
    expect(styles.get('overflow-wrap')?.toString()).toBe('anywhere');
  });
});
