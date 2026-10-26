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

import {describe, expect, it} from 'vitest';
import {
  HTMLProcessingParser,
  loadDefaultJapaneseParser,
  loadDefaultParsers,
  loadDefaultSimplifiedChineseParser,
  loadDefaultThaiParser,
  loadDefaultTraditionalChineseParser,
} from '../index.js';

describe('loadDefaultJapaneseParser', () => {
  it('should return an HTMLProcessingParser instance configured for Japanese', () => {
    const parser = loadDefaultJapaneseParser();
    expect(parser).toBeInstanceOf(HTMLProcessingParser);
    const result = parser.parse('今日は良い天気です');
    expect(result).toEqual(['今日は', '良い', '天気です']);
  });
});

describe('loadDefaultSimplifiedChineseParser', () => {
  it('should return an HTMLProcessingParser instance configured for Simplified Chinese', () => {
    const parser = loadDefaultSimplifiedChineseParser();
    expect(parser).toBeInstanceOf(HTMLProcessingParser);
    const result = parser.parse('简体中文');
    expect(result).toBeDefined();
    expect(result.length).toBeGreaterThan(0);
  });
});

describe('loadDefaultTraditionalChineseParser', () => {
  it('should return an HTMLProcessingParser instance configured for Traditional Chinese', () => {
    const parser = loadDefaultTraditionalChineseParser();
    expect(parser).toBeInstanceOf(HTMLProcessingParser);
    const result = parser.parse('繁體中文');
    expect(result).toBeDefined();
    expect(result.length).toBeGreaterThan(0);
  });
});

describe('loadDefaultThaiParser', () => {
  it('should return an HTMLProcessingParser instance configured for Thai', () => {
    const parser = loadDefaultThaiParser();
    expect(parser).toBeInstanceOf(HTMLProcessingParser);
    const result = parser.parse('วันนี้อากาศดี');
    expect(result).toEqual(['วัน', 'นี้', 'อากาศ', 'ดี']);
  });
});

describe('loadDefaultParsers', () => {
  it('should return a Map with default parsers for supported language codes', () => {
    const parsers = loadDefaultParsers();
    expect(parsers).toBeInstanceOf(Map);
    expect(Array.from(parsers.keys())).toEqual([
      'ja',
      'zh-hans',
      'zh-hant',
      'th',
    ]);

    for (const parser of parsers.values()) {
      expect(parser).toBeInstanceOf(HTMLProcessingParser);
      expect(typeof parser.parse).toBe('function');
    }

    const thParser = parsers.get('th');
    expect(thParser?.parse('วันนี้อากาศดี')).toEqual(['วัน', 'นี้', 'อากาศ', 'ดี']);
  });
});
