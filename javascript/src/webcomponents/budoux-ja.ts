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

import {loadDefaultJapaneseParser, DEFAULT_THRES} from '../parser';

const parser = loadDefaultJapaneseParser();

/**
 * BudouX Japanese Web component.
 */
export class BudouXJapaneseElement extends HTMLElement {
  shadow: ShadowRoot;

  static get observedAttributes() {
    return ['thres'];
  }
  /**
   * BudouX Japanese Web component constructor.
   */
  constructor() {
    super();

    this.shadow = this.attachShadow({mode: 'open'});
    const observer = new MutationObserver(this.sync.bind(this));
    observer.observe(this, {
      attributes: false,
      characterData: true,
      subtree: true,
    });
  }

  connectedCallback() {
    this.sync();
  }

  attributeChangedCallback() {
    this.sync();
  }

  sync() {
    const thres = this.getAttribute('thres');
    this.shadow.innerHTML = parser.translateHTMLString(
      this.innerHTML,
      thres === null ? DEFAULT_THRES : Number(thres)
    );
  }
}

customElements.define('budoux-ja', BudouXJapaneseElement);
