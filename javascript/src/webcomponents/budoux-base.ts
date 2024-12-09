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

import {applyWrapStyle, type HTMLProcessingParser} from '../html_processor.js';

const MUTATION_OBSERVER_OPTIONS = {
  attributes: false,
  characterData: true,
  childList: true,
  subtree: true,
};

/**
 * Base BudouX Web component.
 */
export abstract class BudouXBaseElement extends HTMLElement {
  abstract parser: HTMLProcessingParser;
  observer: MutationObserver;

  /**
   * Base BudouX Web component constructor.
   */
  constructor() {
    super();

    this.observer = new MutationObserver(this.sync.bind(this));
    this.observer.observe(this, MUTATION_OBSERVER_OPTIONS);
  }

  connectedCallback() {
    applyWrapStyle(this);
    this.sync();
  }

  attributeChangedCallback() {
    this.sync();
  }

  sync() {
    this.observer.disconnect();
    this.parser.applyToElement(this);
    this.observer.observe(this, MUTATION_OBSERVER_OPTIONS);
  }
}
