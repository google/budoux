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
import {HTMLProcessor, HTMLProcessorOptions} from '../html_processor.js';

let emulateNotConnected = false;

// Browser compatibilities.
console.assert(!('getComputedStyle' in global));
global.getComputedStyle = (element: Element) => {
  const window = element.ownerDocument.defaultView;
  console.assert(window);
  const style = window!.getComputedStyle(element);
  if (emulateNotConnected) {
    style.display = '';
  } else if (!style.display) {
    // jsdom does not compute unspecified properties.
    const blockify = style.float || style.position;
    style.display = blockify ? 'block' : 'inline';
  }
  return style;
};

const parser = loadDefaultJapaneseParser();

class MockHTMLProcessorBase extends HTMLProcessor {
  constructor(options?: HTMLProcessorOptions) {
    super(parser, options);
  }
}

describe('HTMLProcessor.applyToElement', () => {
  function apply(html: string) {
    const dom = new JSDOM(html);
    const processor = new MockHTMLProcessorBase({
      separator: '/',
      className: 'applied',
    });
    processor.applyToElement(dom.window.document.body);
    return dom.window.document.body.innerHTML;
  }

  for (const test of [
    {
      in: '<div>晴れ</div>',
      out: '<div>晴れ</div>',
    },
    {
      in: '<div>今日は晴れです</div>',
      out: '<div class="applied">今日は/晴れです</div>',
    },
    {
      in: '<div><span>今日は</span>晴れです</div>',
      out: '<div class="applied"><span>今日は</span>/晴れです</div>',
    },
    {
      in: '<div><span>今日は晴れ</span>です</div>',
      out: '<div class="applied"><span>今日は/晴れ</span>です</div>',
    },
    {
      in: '<textarea>今日は晴れです</textarea>',
      out: '<textarea>今日は晴れです</textarea>',
    },
    {
      in: '<div>今日は<code>code</code>晴れです</div>',
      out: '<div class="applied">今日は<code>code</code>/晴れです</div>',
    },
  ]) {
    it(test.in, () => {
      expect(apply(test.in)).toEqual(test.out);
    });
  }
});

describe('HTMLProcessor.applyToElement.separator.node', () => {
  const dom = new JSDOM('<div>今日は良い天気</div>');
  const document = dom.window.document;
  const separator = document.createElement('span');
  separator.style.whiteSpace = 'normal';
  separator.textContent = '\u200B';
  const processor = new MockHTMLProcessorBase({
    separator: separator,
    className: 'applied',
  });
  processor.applyToElement(document.body);
  it('should clone separator element deeply', () => {
    expect(document.body.innerHTML).toEqual(
      '<div class="applied">今日は' +
        '<span style="white-space: normal;">\u200B</span>' +
        '良い天気</div>'
    );
  });
});

describe('HTMLProcessor.getBlocks', () => {
  const getBlocks = (html: string) => {
    const dom = new JSDOM(html);
    const processor = new MockHTMLProcessorBase();
    const blocks = processor.getBlocks(dom.window.document.body);
    const texts = Array.from(
      (function* (blocks) {
        for (const block of blocks)
          yield block.textNodes.map(node => node.nodeValue).join('');
      })(blocks)
    );
    return texts;
  };

  it('should collect all text of a simple block', () => {
    expect(getBlocks('<div>123</div>')).toEqual(['123']);
  });

  it('should collect two blocks separately', () => {
    expect(getBlocks('<div>123</div><div>456</div>')).toEqual(['123', '456']);
  });

  it('should break at <br> elements', () => {
    expect(getBlocks('<div>123<br>456</div>')).toEqual(['123', '456']);
  });

  it('should break at <br> elements inside a span', () => {
    expect(getBlocks('<div>1<span>23<br>45</span>6</div>')).toEqual([
      '123',
      '456',
    ]);
  });

  it('should collect inline boxes as part of the block', () => {
    expect(getBlocks('<div>123<span>456</span>789</div>')).toEqual([
      '123456789',
    ]);
  });

  it('should collect nested blocks separately from the parent block', () => {
    expect(getBlocks('<div>123<div>456</div>789</div>')).toEqual([
      '456',
      '123789',
    ]);
  });

  it('should collect inline-blocks separately from the parent block', () => {
    expect(
      getBlocks('<div>123<div style="display: inline-block">456</div>789</div>')
    ).toEqual(['456', '123789']);
    expect(
      getBlocks(
        '<div>123<span style="display: inline-block">456</span>789</div>'
      )
    ).toEqual(['456', '123789']);
  });

  it('should skip textarea elements', () => {
    expect(getBlocks('<textarea>123</textarea>')).toEqual([]);
  });

  it('should skip <rt> and <rp> elements for <ruby>', () => {
    expect(
      getBlocks('before<ruby>b1<rp>(</rp><rt>r1</rt>b2<rt>r2</rt></ruby>after')
    ).toEqual(['beforeb1b2after']);
  });

  it('should use the built-in rules if the `display` property is empty', () => {
    emulateNotConnected = true;
    expect(getBlocks('<div>123<span>456</span></div>')).toEqual(['123456']);
    expect(getBlocks('<div>123<div>456</div></div>')).toEqual(['456', '123']);
    expect(getBlocks('<div><h1>123</h1><li>456</li></div>')).toEqual([
      '123',
      '456',
    ]);
    emulateNotConnected = false;
  });
});

describe('HTMLProcessor.splitTextNodes', () => {
  class MockText {
    nodeValue: string;

    constructor(text: string) {
      this.nodeValue = text;
    }
  }
  const node123 = new MockText('123') as Text;
  const node456 = new MockText('456') as Text;

  interface NodeAndChunks {
    node: Text;
    chunks: string[];
  }

  class MockHTMLProcessor extends MockHTMLProcessorBase {
    nodeAndChunks: NodeAndChunks[] = [];

    splitTextNode(node: Text, chunks: string[]) {
      this.nodeAndChunks.push({node: node, chunks: chunks});
    }
  }

  function split(nodes: Text[], boundaries: number[]) {
    const processor = new MockHTMLProcessor();
    processor.splitTextNodes(nodes, boundaries);
    return processor.nodeAndChunks;
  }

  it('should not split nodes', () => {
    expect(split([node123], [4])).toEqual([]);
  });

  it('should not split single node at the end', () => {
    expect(split([node123], [3, 4])).toEqual([]);
  });

  it('should not split two nodes at the end', () => {
    expect(split([node123, node456], [6, 7])).toEqual([]);
  });

  it('should split single node at the middle', () => {
    expect(split([node123], [2, 4])).toEqual([
      {node: node123, chunks: ['12', '3']},
    ]);
  });

  it('should split the first node twice', () => {
    expect(split([node123], [1, 2, 4])).toEqual([
      {node: node123, chunks: ['1', '2', '3']},
    ]);
  });

  it('should split the first node at the middle', () => {
    expect(split([node123, node456], [2, 7])).toEqual([
      {node: node123, chunks: ['12', '3']},
    ]);
  });

  it('should split the first node twice', () => {
    expect(split([node123, node456], [1, 2, 7])).toEqual([
      {node: node123, chunks: ['1', '2', '3']},
    ]);
  });

  it('should split the second node at the start', () => {
    expect(split([node123, node456], [3, 7])).toEqual([
      {node: node456, chunks: ['', '456']},
    ]);
  });

  it('should split the second node at the middle', () => {
    expect(split([node123, node456], [5, 7])).toEqual([
      {node: node456, chunks: ['45', '6']},
    ]);
  });

  it('should split the second node twice', () => {
    expect(split([node123, node456], [4, 5, 7])).toEqual([
      {node: node456, chunks: ['4', '5', '6']},
    ]);
  });

  it('should split both nodes at the middle', () => {
    expect(split([node123, node456], [2, 5, 7])).toEqual([
      {node: node123, chunks: ['12', '3']},
      {node: node456, chunks: ['45', '6']},
    ]);
  });

  it('should split both nodes twice', () => {
    expect(split([node123, node456], [1, 2, 4, 5, 7])).toEqual([
      {node: node123, chunks: ['1', '2', '3']},
      {node: node456, chunks: ['4', '5', '6']},
    ]);
  });

  it('should split at every character', () => {
    expect(split([node123, node456], [1, 2, 3, 4, 5, 7])).toEqual([
      {node: node123, chunks: ['1', '2', '3']},
      {node: node456, chunks: ['', '4', '5', '6']},
    ]);
  });
});
