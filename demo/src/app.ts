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

import DOMPurify from 'dompurify';
import { loadDefaultJapaneseParser, loadDefaultSimplifiedChineseParser, loadDefaultTraditionalChineseParser, loadDefaultThaiParser } from 'budoux';

const parsers = new Map([
  ['ja', loadDefaultJapaneseParser()],
  ['zh-hans', loadDefaultSimplifiedChineseParser()],
  ['zh-hant', loadDefaultTraditionalChineseParser()],
  ['th', loadDefaultThaiParser()]
]);
const defaultInputs = new Map([
  ['ja', 'Google の使命は、世界中の情報を<strong>整理</strong>し、<em>世界中の人がアクセス</em>できて使えるようにすることです。'],
  ['zh-hans', '我们的使命是<strong>整合</strong>全球信息，<em>供大众使用</em>，让人人受益。'],
  ['zh-hant', '我們的使命是<strong>匯整</strong>全球資訊，<em>供大眾使用</em>，使人人受惠。'],
  ['th', 'พันธกิจของเราคือการจัดระเบียบข้อมูลในโลกนี้และทำให้เข้าถึงได้ง่ายในทุกที่และมีประโยชน์']
])
const inputTextElement = document.getElementById('input') as HTMLTextAreaElement;
const outputContainerElement = document.getElementById('output') as HTMLElement;
const fontSizeElement = document.getElementById('fontsize') as HTMLInputElement;
const brCheckElement = document.getElementById('wbr2br') as HTMLInputElement;
const modelSelectElement = document.getElementById('model') as HTMLSelectElement;
const url = new URL(document.location.href);
const worker = new Worker('./worker.js');
worker.onmessage = (e: MessageEvent) => {
  console.log('response from worker:', e);
};


/**
 * Runs the BudouX model to process the input text and render the processed HTML.
 */
const run = () => {
  outputContainerElement.innerHTML = DOMPurify.sanitize(inputTextElement.value);
  const model = modelSelectElement.value;
  worker.postMessage({'sentence': outputContainerElement.textContent, 'model': model});
  const parser = parsers.get(model);
  if (!parser) return;
  parser.applyToElement(outputContainerElement);
  const renderWithBR = brCheckElement.checked;
  if (renderWithBR) {
    outputContainerElement.innerHTML = DOMPurify.sanitize(
      outputContainerElement.innerHTML.replace(/\u200b/g, '<br>'));
  }
  url.searchParams.set('q', inputTextElement.value);
  window.history.replaceState('', '', url.toString());
};

/**
 * Initializes the app.
 */
const init = () => {
  const lang = url.searchParams.get('lang');
  if (lang) modelSelectElement.value = lang;
  const input = url.searchParams.get('q') || defaultInputs.get(modelSelectElement.value);
  if (input) inputTextElement.value = input;
  run();
}

fontSizeElement.addEventListener('input', () => {
  outputContainerElement.style.fontSize = `${fontSizeElement.value}rem`;
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