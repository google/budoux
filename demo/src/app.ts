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

import { loadDefaultJapaneseParser } from 'budoux';

const parser = loadDefaultJapaneseParser();
const inputTextElement = document.getElementById('input') as HTMLTextAreaElement;
const outputContainerElement = document.getElementById('output') as HTMLElement;
const fontSizeElement = document.getElementById('fontsize') as HTMLInputElement;
const thresholdElement = document.getElementById('threshold') as HTMLInputElement;
const brCheckElement = document.getElementById('wbr2br') as HTMLInputElement;


declare global {
  interface Window {
    DOMPurify: {
      sanitize: (value: string) => string;
    }
  }
}

const run = () => {
  outputContainerElement.innerHTML = window.DOMPurify.sanitize(inputTextElement.value);
  const threshold = Number(thresholdElement.value);
  parser.applyElement(outputContainerElement, threshold);
  outputContainerElement.style.fontSize = `${fontSizeElement.value}rem`;
  const renderWithBR = brCheckElement.checked;
  if (renderWithBR) {
    outputContainerElement.innerHTML = window.DOMPurify.sanitize(
      outputContainerElement.innerHTML.replace(/<wbr>/g, '<br>'));
  }
};

fontSizeElement.addEventListener('input', () => {
  run();
})

inputTextElement.addEventListener('input', () => {
  run();
});

thresholdElement.addEventListener('input', () => {
  run();
});

brCheckElement.addEventListener('input', () => {
  run();
})

run();
