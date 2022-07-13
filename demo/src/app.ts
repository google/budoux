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

import { loadDefaultJapaneseParser, loadDefaultSimplifiedChineseParser } from 'budoux';

const parsers = new Map([
  ['ja', loadDefaultJapaneseParser()],
  ['zh-hans', loadDefaultSimplifiedChineseParser()],
]);
const defaultInputs = new Map([
  ['ja', 'Google の使命は、世界中の情報を<strong>整理</strong>し、<em>世界中の人がアクセス</em>できて使えるようにすることです。'],
  ['zh-hans', '我们的使命是<strong>整合</strong>全球信息，<em>供大众使用</em>，让人人受益。'],
])
const inputTextElement = document.getElementById('input') as HTMLTextAreaElement;
const outputContainerElement = document.getElementById('output') as HTMLElement;
const fontSizeElement = document.getElementById('fontsize') as HTMLInputElement;
const brCheckElement = document.getElementById('wbr2br') as HTMLInputElement;
const modelSelectElement = document.getElementById('model') as HTMLSelectElement;
const url = new URL(document.location.href);


declare global {
  interface Window {
    DOMPurify: {
      sanitize: (value: string) => string;
    }
  }
}

/**
 * Runs the BudouX model to process the input text and render the processed HTML.
 */
const run = () => {
  outputContainerElement.innerHTML = window.DOMPurify.sanitize(inputTextElement.value);
  const model = modelSelectElement.value;
  const parser = parsers.get(model);
  if (!parser) return;
  parser.applyElement(outputContainerElement);
  outputContainerElement.style.fontSize = `${fontSizeElement.value}rem`;
  const renderWithBR = brCheckElement.checked;
  if (renderWithBR) {
    outputContainerElement.innerHTML = window.DOMPurify.sanitize(
      outputContainerElement.innerHTML.replace(/<wbr>/g, '<br>'));
  }
};

/**
 * Initializes the app.
 */
const init = () => {
  const lang = url.searchParams.get('lang');
  if (lang) modelSelectElement.value = lang;
  const input = defaultInputs.get(modelSelectElement.value);
  if (input) inputTextElement.value = input;
  run();
}

fontSizeElement.addEventListener('input', () => {
  run();
})

inputTextElement.addEventListener('input', () => {
  run();
});

brCheckElement.addEventListener('input', () => {
  run();
});

modelSelectElement.addEventListener('change', () => {
  url.searchParams.set('lang', modelSelectElement.value);
  window.history.pushState('', '', url.toString());
  const input = defaultInputs.get(modelSelectElement.value);
  if (input) inputTextElement.value = input;
  run();
})

init();