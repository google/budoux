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

import {loadDefaultJapaneseParser} from '../index.js';
import {
  HTMLProcessingParser,
  HTMLProcessor,
  HTMLProcessorOptions,
  NodeOrTextForTesting,
  ParagraphForTesting,
} from '../html_processor.js';
import {parseFromString} from '../dom.js';
import {setInnerHtml, createDocument, isBrowser} from './testutils.js';

const parser = loadDefaultJapaneseParser();

class MockHTMLProcessorBase extends HTMLProcessor {
  constructor(options?: HTMLProcessorOptions) {
    super(parser, options);
  }
}

function isEqualNodeWithStyleNormalized(a: HTMLElement, b: HTMLElement) {
  if (a.nodeType !== b.nodeType) return false;
  if (a.nodeType === 3) {
    // is a text node
    return a.nodeValue === b.nodeValue;
  }
  if (a.nodeName !== b.nodeName) return false;
  if (a.attributes.length !== b.attributes.length) return false;
  for (let i = 0; i < a.attributes.length; i++) {
    const attribName = a.attributes[i].name;
    const aAttribValue = a.getAttribute(attribName);
    const bAttribValue = b.getAttribute(attribName);
    if (attribName === 'style') {
      const aNormalizedStyle = aAttribValue
        ?.replace(/\s+/g, '')
        .replace(/;$/, '');
      const bNormalizedStyle = bAttribValue
        ?.replace(/\s+/g, '')
        .replace(/;$/, '');
      if (aNormalizedStyle !== bNormalizedStyle) return false;
    } else {
      if (aAttribValue !== bAttribValue) return false;
    }
  }
  if (a.childNodes.length !== b.childNodes.length) return false;
  for (let i = 0; i < a.childNodes.length; i++) {
    if (
      !isEqualNodeWithStyleNormalized(
        a.childNodes[i] as HTMLElement,
        b.childNodes[i] as HTMLElement
      )
    ) {
      return false;
    }
  }
  return true;
}

function getBlocks(html: string): IterableIterator<ParagraphForTesting> {
  const document = createDocument();
  setInnerHtml(document.body, html);
  const processor = new MockHTMLProcessorBase();
  return processor.getBlocks(document.body);
}

describe('HTMLProcessor.applyToElement', () => {
  const document = createDocument();
  const wbr = document.createElement('wbr');
  function apply(html: string, separator: string | Node) {
    setInnerHtml(document.body, html);
    const processor = new MockHTMLProcessorBase({
      separator: separator,
      className: 'applied',
    });
    processor.applyToElement(document.body);
    return document.body.innerHTML;
  }

  for (const test of [
    {
      in: '<div>晴れ</div>',
      out: '<div>晴れ</div>',
      browserOnly: false,
    },
    {
      in: '<div>今日は晴れです</div>',
      out: '<div class="applied">今日は|晴れです</div>',
      browserOnly: false,
    },
    {
      in: '<div><span>今日は</span>晴れです</div>',
      out: '<div class="applied"><span>今日は</span>|晴れです</div>',
      browserOnly: false,
    },
    {
      in: '<div><span>今日は晴れ</span>です</div>',
      out: '<div class="applied"><span>今日は|晴れ</span>です</div>',
      browserOnly: false,
    },
    {
      in: '<code>今日は晴れです</code>',
      out: '<code>今日は晴れです</code>',
      browserOnly: false,
    },
    {
      in: '<div>今日は<code>code</code>晴れです</div>',
      out: '<div class="applied">今日は<code>code</code>|晴れです</div>',
      browserOnly: false,
    },
    {
      in: '<div>今日は晴れ、今日は晴れ</div>',
      out: '<div class="applied">今日は|晴れ、|今日は|晴れ</div>',
      browserOnly: false,
    },
    {
      in: '<div>今日は<nobr>晴れ、今日は</nobr>晴れ</div>',
      out: '<div class="applied">今日は|<nobr>晴れ、今日は</nobr>|晴れ</div>',
      browserOnly: true,
    },
    {
      in: '<div>今日は<span style="white-space: nowrap">晴れ、今日は</span>晴れ</div>',
      out: '<div class="applied">今日は|<span style="white-space: nowrap">晴れ、今日は</span>|晴れ</div>',
      browserOnly: true,
    },
  ]) {
    // Test when the separator is an `Element`.
    it(test.in, () => {
      const out = test.out.replace(/\|/g, '<wbr>');
      if (test.browserOnly && !isBrowser) return;
      const result = apply(test.in, wbr);
      const isEqual = isEqualNodeWithStyleNormalized(
        parseFromString(result).body,
        parseFromString(out).body
      );
      if (!isEqual) {
        console.log('actual:', result);
        console.log('expected:', out);
      }
      expect(isEqual).toBeTrue();
    });
    // Test when the separator is a `string`.
    it(test.in, () => {
      const out = test.out.replace(/\|/g, '/');
      if (test.browserOnly && !isBrowser) return;
      const result = apply(test.in, '/');
      const isEqual = isEqualNodeWithStyleNormalized(
        parseFromString(result).body,
        parseFromString(out).body
      );
      if (!isEqual) {
        console.log('actual:', result);
        console.log('expected:', out);
      }
      expect(isEqual).toBeTrue();
    });
  }
});

describe('HTMLProcessor.applyToElement.separator.node', () => {
  it('should clone separator element deeply', () => {
    const doc = createDocument();
    setInnerHtml(doc.body, '<div>今日は良い天気です</div>');
    const separator = doc.createElement('span');
    separator.style.whiteSpace = 'normal';
    separator.textContent = '\u200B';
    const processor = new MockHTMLProcessorBase({
      separator: separator,
      className: 'applied',
    });
    processor.applyToElement(doc.body);
    const expected =
      '<div class="applied">今日は' +
      '<span style="white-space: normal;">\u200B</span>良い' +
      '<span style="white-space: normal;">\u200B</span>天気です</div>';
    const isEqual = isEqualNodeWithStyleNormalized(
      doc.body,
      parseFromString(expected).body
    );
    if (!isEqual) {
      console.log('actual:', doc.body.innerHTML);
      console.log('expected:', expected);
    }
    expect(isEqual).toBeTrue();
  });
});

describe('HTMLProcessor.getBlocks', () => {
  function getText(html: string): string[] {
    const blocks = getBlocks(html);
    return Array.from(
      (function* (blocks) {
        for (const block of blocks) yield block.text;
      })(blocks)
    );
  }

  it('should collect all text of a simple block', () => {
    expect(getText('<div>123</div>')).toEqual(['123']);
  });

  it('should collect two blocks separately', () => {
    expect(getText('<div>123</div><div>456</div>')).toEqual(['123', '456']);
  });

  it('should break at <br> elements', () => {
    expect(getText('<div>123<br>456</div>')).toEqual(['123', '456']);
  });

  it('should break at <br> elements inside a span', () => {
    expect(getText('<div>1<span>23<br>45</span>6</div>')).toEqual([
      '123',
      '456',
    ]);
  });

  it('should collect inline boxes as part of the block', () => {
    expect(getText('<div>123<span>456</span>789</div>')).toEqual(['123456789']);
  });

  it('should collect nested blocks separately from the parent block', () => {
    expect(getText('<div>123<div>456</div>789</div>')).toEqual([
      '456',
      '123789',
    ]);
  });

  it('should collect inline-blocks separately from the parent block', () => {
    expect(
      getText('<div>123<div style="display: inline-block">456</div>789</div>')
    ).toEqual(['456', '123789']);
    if (isBrowser) {
      expect(
        getText(
          '<div>123<span style="display: inline-block">456</span>789</div>'
        )
      ).toEqual(['456', '123789']);
    }
  });

  it('should skip textarea elements', () => {
    expect(getText('<textarea>123</textarea>')).toEqual([]);
  });

  it('should skip <rt> and <rp> elements for <ruby>', () => {
    expect(
      getText('before<ruby>b1<rp>(</rp><rt>r1</rt>b2<rt>r2</rt></ruby>after')
    ).toEqual(['beforeb1b2after']);
  });

  it('should use the built-in rules if the `display` property is empty', () => {
    expect(getText('<div>123<span>456</span></div>')).toEqual(['123456']);
    expect(getText('<div>123<div>456</div></div>')).toEqual(['456', '123']);
    expect(getText('<div><h1>123</h1><li>456</li></div>')).toEqual([
      '123',
      '456',
    ]);
  });
});

describe('HTMLProcessor.forcedOpportunities', () => {
  function forcedOpportunities(html: string) {
    const blocks = getBlocks(html);
    return Array.from(
      (function* (blocks) {
        for (const block of blocks) {
          yield {
            indices: block.getForcedOpportunities(),
            after: block.nodes.map(block => block.hasBreakOpportunityAfter),
          };
        }
      })(blocks)
    );
  }

  it('<wbr> should set has_break_opportunity_after', () => {
    expect(forcedOpportunities('123<wbr>456')).toEqual([
      {indices: [3], after: [true, false]},
    ]);
  });
  it('Nested <wbr> should set has_break_opportunity_after', () => {
    expect(forcedOpportunities('123<span><wbr></span>456')).toEqual([
      {indices: [3], after: [true, false]},
    ]);
  });
  it('ZWSP should be in forcedOpportunities', () => {
    expect(forcedOpportunities('123<span>\u200B456</span>')).toEqual([
      {indices: [4], after: [false, false]},
    ]);
  });
});

describe('HTMLProcessor.splitNodes', () => {
  class MockNode extends NodeOrTextForTesting {
    constructor(text: string) {
      super(text);
    }

    clear() {
      this.chunks = [];
    }

    override get canSplit(): boolean {
      return true;
    }

    override split() {}
  }
  const node123 = new MockNode('123');
  const node456 = new MockNode('456');

  function split(nodes: MockNode[], boundaries: number[]): string[][] {
    for (const node of nodes) {
      node.clear();
    }
    const processor = new MockHTMLProcessorBase();
    processor.splitNodes(nodes, boundaries);
    const result = nodes.map(node => node.chunks);
    return result;
  }

  it('should not split nodes', () => {
    expect(split([node123], [4])).toEqual([[]]);
  });

  it('should not split single node at the end', () => {
    expect(split([node123], [3, 4])).toEqual([[]]);
  });

  it('should not split two nodes at the end', () => {
    expect(split([node123, node456], [6, 7])).toEqual([[], []]);
  });

  it('should split single node at the middle', () => {
    expect(split([node123], [2, 4])).toEqual([['12', '3']]);
  });

  it('should split the first node twice', () => {
    expect(split([node123], [1, 2, 4])).toEqual([['1', '2', '3']]);
  });

  it('should split the first node at the middle', () => {
    expect(split([node123, node456], [2, 7])).toEqual([['12', '3'], []]);
  });

  it('should split the first node twice', () => {
    expect(split([node123, node456], [1, 2, 7])).toEqual([['1', '2', '3'], []]);
  });

  it('should split the second node at the start', () => {
    expect(split([node123, node456], [3, 7])).toEqual([[], ['', '456']]);
  });

  it('should split the second node at the middle', () => {
    expect(split([node123, node456], [5, 7])).toEqual([[], ['45', '6']]);
  });

  it('should split the second node twice', () => {
    expect(split([node123, node456], [4, 5, 7])).toEqual([[], ['4', '5', '6']]);
  });

  it('should split both nodes at the middle', () => {
    expect(split([node123, node456], [2, 5, 7])).toEqual([
      ['12', '3'],
      ['45', '6'],
    ]);
  });

  it('should split both nodes twice', () => {
    expect(split([node123, node456], [1, 2, 4, 5, 7])).toEqual([
      ['1', '2', '3'],
      ['4', '5', '6'],
    ]);
  });

  it('should split at every character', () => {
    expect(split([node123, node456], [1, 2, 3, 4, 5, 7])).toEqual([
      ['1', '2', '3'],
      ['', '4', '5', '6'],
    ]);
  });
});

describe('HTMLProcessingParser.applyToElement', () => {
  const checkEqual = (
    model: {[key: string]: {[key: string]: number}},
    inputHTML: string,
    expectedHTML: string
  ) => {
    const inputDOM = parseFromString(inputHTML);
    const inputDocument = inputDOM.querySelector('p') as unknown as HTMLElement;
    const parser = new HTMLProcessingParser(model);
    parser.applyToElement(inputDocument);
    const expectedDocument = parseFromString(expectedHTML);
    const expectedElement = expectedDocument.querySelector(
      'p'
    ) as unknown as HTMLElement;
    const isEqual = isEqualNodeWithStyleNormalized(
      inputDocument,
      expectedElement
    );
    if (!isEqual) {
      console.log('actual:', inputDocument.outerHTML);
      console.log('expected:', expectedElement.outerHTML);
    }
    expect(isEqual).toBeTrue();
  };
  const style = 'word-break: keep-all; overflow-wrap: anywhere;';

  it('should insert ZWSPs where the sentence should break.', () => {
    const inputHTML = '<p>xyzabcabc</p>';
    const expectedHTML = `<p style="${style}">xyz\u200Babc\u200Babc</p>`;
    const model = {
      UW4: {a: 1001}, // means "should separate right before 'a'".
    };
    checkEqual(model, inputHTML, expectedHTML);
  });

  it('should insert ZWSPs even it overlaps with other HTML tags.', () => {
    const inputHTML = '<p>xy<a href="#">zabca</a>bc</p>';
    const expectedHTML = `<p style="${style}">xy<a href="#">z\u200Babc\u200Ba</a>bc</p>`;
    const model = {
      UW4: {a: 1001}, // means "should separate right before 'a'".
    };
    checkEqual(model, inputHTML, expectedHTML);
  });

  it('should not insert ZWSPs to where input has WBR tags already.', () => {
    const inputHTML = '<p>xyz<wbr>abcabc</p>';
    const expectedHTML = `<p style="${style}">xyz<wbr>abc\u200Babc</p>`;
    const model = {
      UW4: {a: 1001}, // means "should separate right before 'a'".
    };
    checkEqual(model, inputHTML, expectedHTML);
  });
  it('should not insert ZWSPs to where input has ZWSPs.', () => {
    const inputHTML = '<p>xyz\u200Babcabc</p>';
    const expectedHTML = `<p style="${style}">xyz\u200babc\u200Babc</p>`;
    const model = {
      UW4: {a: 1001}, // means "should separate right before 'a'".
    };
    checkEqual(model, inputHTML, expectedHTML);
  });
});

describe('HTMLProcessingParser.translateHTMLString', () => {
  const defaultModel = {
    UW4: {a: 1001}, // means "should separate right before 'a'".
  };
  const checkEqual = (
    model: {[key: string]: {[key: string]: number}},
    inputHTML: string,
    expectedHTML: string
  ) => {
    const parser = new HTMLProcessingParser(model);
    const result = parser.translateHTMLString(inputHTML);
    const resultDocument = parseFromString(result);
    const expectedDocument = parseFromString(expectedHTML);
    const isEqual = isEqualNodeWithStyleNormalized(
      resultDocument.body,
      expectedDocument.body
    );
    if (!isEqual) {
      console.log('actual', result);
      console.log('expected', expectedHTML);
    }
    expect(isEqual).toBeTrue();
  };

  it('should output a html string with a SPAN parent with proper style attributes.', () => {
    const inputHTML = 'xyzabcd';
    const expectedHTML = `<span
    style="word-break: keep-all; overflow-wrap: anywhere;">xyz\u200Babcd</span>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });

  it('should not add a SPAN parent if the input already has one single parent.', () => {
    const inputHTML = '<p class="foo" style="color: red">xyzabcd</p>';
    const expectedHTML = `<p class="foo"
       style="color: red; word-break: keep-all; overflow-wrap: anywhere;"
    >xyz\u200Babcd</p>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });

  it('should return a blank string if the input is blank.', () => {
    const inputHTML = '';
    const expectedHTML = '';
    checkEqual({}, inputHTML, expectedHTML);
  });

  it('should pass script tags as-is.', () => {
    const inputHTML = 'xyz<script>alert(1);</script>xyzabc';
    const expectedHTML = `<span
    style="word-break: keep-all; overflow-wrap: anywhere;"
    >xyz<script>alert(1);</script>xyz\u200Babc</span>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });

  it('should skip some specific tags.', () => {
    const inputHTML = 'xyz<code>abc</code>abc';
    const expectedHTML = `<span
    style="word-break: keep-all; overflow-wrap: anywhere;"
    >xyz<code>abc</code>\u200Babc</span>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });

  it('should not ruin attributes of child elements.', () => {
    const inputHTML = 'xyza<a href="#" hidden>bc</a>abc';
    const expectedHTML = `<span
    style="word-break: keep-all; overflow-wrap: anywhere;"
    >xyz\u200Ba<a href="#" hidden>bc</a>\u200Babc</span>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });

  it('should work with emojis.', () => {
    const inputHTML = 'xyza🇯🇵🇵🇹abc';
    const expectedHTML = `<span
    style="word-break: keep-all; overflow-wrap: anywhere;"
    >xyz\u200Ba🇯🇵🇵🇹\u200Babc</span>`;
    checkEqual(defaultModel, inputHTML, expectedHTML);
  });
});
