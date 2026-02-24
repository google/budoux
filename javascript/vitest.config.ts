/**
 * @license
 * Copyright 2026 Google LLC
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

import {playwright} from '@vitest/browser-playwright';
import path from 'path';
import {fileURLToPath} from 'url';
import {defineConfig} from 'vitest/config';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          environment: 'node',
          include: [
            'src/tests/test_cli.ts',
            'src/tests/test_parser.ts',
            'src/tests/test_html_processor.ts',
          ],
        },
      },
      {
        test: {
          browser: {
            enabled: true,
            provider: playwright(),
            headless: true,
            instances: [{browser: 'chromium'}],
          },
          include: [
            'src/tests/test_webcomponents.ts',
            'src/tests/test_parser.ts',
            'src/tests/test_html_processor.ts',
          ],
        },
        resolve: {
          alias: [
            // Transformations described in package.json's "browser" field.
            {
              find: /^(\.)?\.\/dom\.js$/,
              replacement: path.resolve(__dirname, 'src/dom-browser.ts'),
            },
            {
              find: /^(\.)?\.\/testutils.js$/,
              replacement: path.resolve(
                __dirname,
                'src/tests/testutils-browser.ts'
              ),
            },
          ],
        },
      },
    ],
  },
});
