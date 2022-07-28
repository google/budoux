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

import {Parser} from './parser.js';

const assert = console.assert;

const ZWSP = '\u200B'; // U+200B ZERO WIDTH SPACE

// We could use `Node.TEXT_NODE` and `Node.ELEMENT_NODE` in a browser context,
// but we define the same here for Node.js environments.
const NodeType = {
  ELEMENT_NODE: 1,
  TEXT_NODE: 3,
};

const DomAction = {
  Inline: 0,
  Block: 1,
  Skip: 2,
  Break: 3,
} as const;
type DomAction = typeof DomAction[keyof typeof DomAction];

/**
 * Determines the action from an element name, as defined in
 * {@link https://html.spec.whatwg.org/multipage/rendering.html HTML Rendering}.
 * See also {@link actionForElement}.
 */
const domActions: {[name: string]: DomAction} = {
  // Hidden elements
  // https://html.spec.whatwg.org/multipage/rendering.html#hidden-elements
  AREA: DomAction.Skip,
  BASE: DomAction.Skip,
  BASEFONT: DomAction.Skip,
  DATALIST: DomAction.Skip,
  HEAD: DomAction.Skip,
  LINK: DomAction.Skip,
  META: DomAction.Skip,
  NOEMBED: DomAction.Skip,
  NOFRAMES: DomAction.Skip,
  PARAM: DomAction.Skip,
  RP: DomAction.Skip,
  SCRIPT: DomAction.Skip,
  STYLE: DomAction.Skip,
  TEMPLATE: DomAction.Skip,
  TITLE: DomAction.Skip,
  NOSCRIPT: DomAction.Skip,

  // Flow content
  // https://html.spec.whatwg.org/multipage/rendering.html#flow-content-3
  HR: DomAction.Break,
  // Disable if `white-space: pre`.
  LISTING: DomAction.Skip,
  PLAINTEXT: DomAction.Skip,
  PRE: DomAction.Skip,
  XMP: DomAction.Skip,

  // Phrasing content
  // https://html.spec.whatwg.org/multipage/rendering.html#phrasing-content-3
  BR: DomAction.Break,
  RT: DomAction.Skip,

  // Form controls
  // https://html.spec.whatwg.org/multipage/rendering.html#form-controls
  INPUT: DomAction.Skip,
  SELECT: DomAction.Skip,
  BUTTON: DomAction.Skip,
  TEXTAREA: DomAction.Skip,

  // Other elements where the phrase-based line breaking should be disabled.
  // https://github.com/google/budoux/blob/main/budoux/skip_nodes.json
  ABBR: DomAction.Skip,
  CODE: DomAction.Skip,
  IFRAME: DomAction.Skip,
  TIME: DomAction.Skip,
  VAR: DomAction.Skip,
};

const defaultBlockElements = new Set([
  // 15.3.2 The page
  'HTML',
  'BODY',
  // 15.3.3 Flow content
  'ADDRESS',
  'BLOCKQUOTE',
  'CENTER',
  'DIALOG',
  'DIV',
  'FIGURE',
  'FIGCAPTION',
  'FOOTER',
  'FORM',
  'HEADER',
  'LEGEND',
  'LISTING',
  'MAIN',
  'P',
  // 15.3.6 Sections and headings
  'ARTICLE',
  'ASIDE',
  'H1',
  'H2',
  'H3',
  'H4',
  'H5',
  'H6',
  'HGROUP',
  'NAV',
  'SECTION',
  // 15.3.7 Lists
  'DIR',
  'DD',
  'DL',
  'DT',
  'MENU',
  'OL',
  'UL',
  'LI',
  // 15.3.8 Tables
  'TABLE',
  'CAPTION',
  'COL',
  'TR',
  'TD',
  'TH',
  // 15.3.12 The fieldset and legend elements
  'FIELDSET',
  // 15.5.4 The details and summary elements
  'DETAILS',
  'SUMMARY',
  // 15.5.12 The marquee element
  'MARQUEE',
]);

/**
 * Determine the action for an element.
 * @param element An element to determine the action for.
 * @returns The {@link domActions} for the element.
 */
function actionForElement(element: Element): DomAction {
  const nodeName = element.nodeName;
  const action = domActions[nodeName];
  if (action !== undefined) return action;

  if (typeof getComputedStyle === 'function') {
    const style = getComputedStyle(element);
    switch (style.whiteSpace) {
      case 'nowrap':
      case 'pre':
        return DomAction.Skip;
    }

    const display = style.display;
    if (display)
      return display === 'inline' ? DomAction.Inline : DomAction.Block;
    // `display` is an empty string if the element is not connected.
  }
  // Use the built-in rules if the `display` property is empty, or if
  // `getComputedStyle` is missing (e.g., jsdom.)
  return defaultBlockElements.has(nodeName)
    ? DomAction.Block
    : DomAction.Inline;
}

/**
 * Represents a "paragraph", broken by block boundaries or forced breaks.
 *
 * A CSS
 * {@link https://drafts.csswg.org/css2/#inline-formatting inline formatting context}
 * is usually a "paragraph", but it can be broken into multiple paragraphs by
 * forced breaks such as `<br>`.
 */
class Paragraph {
  element: HTMLElement;
  textNodes: Text[] = [];

  constructor(element: HTMLElement) {
    this.element = element;
  }

  hasText(): boolean {
    return this.textNodes.length > 0;
  }
}

/**
 * Options for {@link HTMLProcessor}.
 */
export interface HTMLProcessorOptions {
  /**
   * This class name is added to the containing block
   * when the BudouX is applied.
   *
   * The caller is responsible for defining the class.
   * {@link defineClassAs} can append a `<style>` element
   * that defines the default styles as a class.
   *
   * When falsy, an inline style is set instead.
   */
  className?: string;
  /**
   * The separator to insert at each semantics boundary.
   *
   * When it's a {@link Node}, a clone of the {@link Node} will be inserted.
   *
   * The default value is U+200B ZERO WIDTH SPACE.
   */
  separator?: string | Node;
}

/**
 * Applies the BudouX to the given DOM.
 *
 * This class has following advantages over
 * {@link Parser.applyElement}.
 * * It recognizes paragraphs and applies the BudouX for each
 *   paragraph separately.
 * * It can customize how to insert break opportunities.
 *   See {@link separator} for more details.
 * * It is generally faster and more memory efficient, but the
 *   code size is larger.
 */
export class HTMLProcessor {
  private parser_: Parser;
  /** See {@link HTMLProcessorOptions.className}. */
  className?: string;
  /** See {@link HTMLProcessorOptions.separator}. */
  separator: string | Node = ZWSP;

  /**
   * @param parser A BudouX {@link Parser} to compute semantic line breaks.
   */
  constructor(parser: Parser, options?: HTMLProcessorOptions) {
    this.parser_ = parser;
    if (options !== undefined) {
      if (options.className !== undefined) this.className = options.className;
      if (options.separator !== undefined) this.separator = options.separator;
    }
  }

  /**
   * Applies markups for semantic line breaks to the given HTML element.
   *
   * It breaks descendant nodes into paragraphs,
   * and applies the BudouX to each paragraph.
   * @param element The input element.
   */
  applyToElement(element: HTMLElement) {
    for (const block of this.getBlocks(element)) {
      assert(block.hasText());
      this.applyToParagraph(block);
    }
  }

  /**
   * Find paragraphs from a given HTML element.
   * @param element The root element to find paragraphs.
   * @param parent The parent {@link Paragraph} if any.
   * @returns A list of {@link Paragraph}s.
   */
  *getBlocks(
    element: HTMLElement,
    parent?: Paragraph
  ): IterableIterator<Paragraph> {
    assert(element.nodeType === NodeType.ELEMENT_NODE);

    // Skip if it was once applied to this element.
    if (this.className && element.classList.contains(this.className)) return;

    const action = actionForElement(element);
    if (action === DomAction.Skip) return;

    if (action === DomAction.Break) {
      if (parent && parent.hasText()) {
        yield parent;
        parent.textNodes = [];
      }
      assert(!element.firstChild);
      return;
    }

    // Determine if this element creates a new inline formatting context, or if
    // this element belongs to the parent inline formatting context.
    assert(action === DomAction.Block || action === DomAction.Inline);
    const isNewBlock = !parent || action === DomAction.Block;
    const block = isNewBlock ? new Paragraph(element) : parent;
    assert(block);

    // Collect all text nodes in this inline formatting context, while searching
    // descendant elements recursively.
    for (const child of element.childNodes) {
      switch (child.nodeType) {
        case NodeType.ELEMENT_NODE:
          for (const childBlock of this.getBlocks(child as HTMLElement, block))
            yield childBlock;
          break;
        case NodeType.TEXT_NODE:
          block.textNodes.push(child as Text);
          break;
      }
    }

    // Apply if this is an inline formatting context.
    if (isNewBlock && block.hasText()) yield block;
  }

  /**
   * Apply the BudouX to the given {@link Paragraph}.
   * @param paragraph The {@link Paragraph} to apply.
   */
  applyToParagraph(paragraph: Paragraph): void {
    const textNodes = paragraph.textNodes;
    assert(textNodes.length > 0);
    const texts = textNodes.map(node => node.nodeValue);
    const text = texts.join('');
    // No changes if whitespace-only.
    if (/^\s*$/.test(text)) return;

    // Split the text into a list of phrases.
    const phrases = this.parser_.parse(text);
    assert(phrases.length > 0);
    assert(
      phrases.reduce((sum, phrase) => sum + phrase.length, 0) === text.length
    );
    // No changes if single phrase.
    if (phrases.length <= 1) return;

    // Compute the boundary indices from the list of phrase strings.
    const boundaries = [];
    let char_index = 0;
    for (const phrase of phrases) {
      assert(phrase.length > 0);
      char_index += phrase.length;
      boundaries.push(char_index);
    }

    // The break opportunity at the end of a block is not needed. Instead of
    // removing it, turn it to a sentinel for `splitTextNodesAtBoundaries` by
    // making it larger than the text length.
    assert(boundaries[0] > 0);
    assert(boundaries[boundaries.length - 1] === text.length);
    ++boundaries[boundaries.length - 1];
    assert(boundaries.length > 1);

    this.splitTextNodes(textNodes, boundaries);
    this.applyBlockStyle(paragraph.element);
  }

  /**
   * Split {@link Text} nodes at the specified boundaries.
   * @param textNodes A list of {@link Text}.
   * @param boundaries A list of indices of the text to split at.
   */
  splitTextNodes(textNodes: Text[], boundaries: number[]): void {
    assert(boundaries.length > 0);
    const textLen = textNodes.reduce(
      (sum, node) => sum + (node.nodeValue ? node.nodeValue.length : 0),
      0
    );
    // The last boundary must be a sentinel.
    assert(boundaries[boundaries.length - 1] > textLen);

    let boundary_index = 0;
    let boundary = boundaries[0];
    assert(boundary > 0);
    let nodeStart = 0; // the start index of the `nodeText` in the whole text.
    for (const node of textNodes) {
      const nodeText = node.nodeValue;
      if (!nodeText) continue;

      // Check if the next boundary is in this `node`.
      const nodeEnd = nodeStart + nodeText.length;
      if (boundary >= nodeEnd) {
        nodeStart = nodeEnd;
        continue;
      }

      // Compute the boundary indices in the `nodeText`.
      const chunks = [];
      let chunkStartInNode = 0;
      while (boundary < nodeEnd) {
        const boundaryInNode = boundary - nodeStart;
        assert(boundaryInNode >= chunkStartInNode);
        chunks.push(nodeText.substring(chunkStartInNode, boundaryInNode));
        chunkStartInNode = boundaryInNode;
        ++boundary_index;
        assert(boundaries[boundary_index] > boundary);
        boundary = boundaries[boundary_index];
      }
      assert(chunks.length > 0);

      // Add the rest of the `nodeText` and split the `node`.
      if (chunkStartInNode < nodeText.length)
        chunks.push(nodeText.substring(chunkStartInNode));
      this.splitTextNode(node, chunks);

      nodeStart = nodeEnd;
    }

    // Check if all nodes and boundaries are consumed.
    assert(nodeStart === textLen);
    assert(boundary_index < boundaries.length);
    assert(boundaries[boundary_index] >= textLen);
  }

  /**
   * Split a {@link Text} node in the same way as the given chunks.
   * @param node A {@link Text} node to split.
   * @param chunks A list of {@link string} specifying where to split.
   * Joining all {@link chunks} must be equal to {@link node.nodeValue}.
   */
  splitTextNode(node: Text, chunks: string[]): void {
    assert(chunks.length > 1);
    assert(node.nodeValue === chunks.join(''));

    const separator = this.separator;
    if (typeof separator === 'string') {
      // If the `separator` is a string, insert it at each boundary.
      node.nodeValue = chunks.join(separator);
      return;
    }

    // Otherwise create a `Text` node for each chunk, with the separator node
    // between them, and replace the `node` with them.
    const document = node.ownerDocument;
    let nodes = [];
    for (const chunk of chunks) {
      if (chunk) nodes.push(document.createTextNode(chunk));
      nodes.push(null);
    }
    nodes.pop();
    nodes = nodes.map(n => (n ? n : separator.cloneNode(true)));
    node.replaceWith(...nodes);
  }

  /**
   * Applies the block style to the given element.
   * @param element The element to apply the block style.
   */
  applyBlockStyle(element: HTMLElement): void {
    if (this.className) {
      element.classList.add(this.className);
      return;
    }

    const style = element.style;
    style.wordBreak = 'keep-all';
    style.overflowWrap = 'break-word';
  }

  /**
   * Append a `<style>` element that defines the default styles as a class.
   * @param document The document to append to.
   * @param className The CSS class name.
   */
  static defineClassAs(document: Document, className: string): void {
    const style = document.createElement('style');
    style.textContent = `.${className} { word-break: keep-all; overflow-wrap: break-word; }`;
    document.head.appendChild(style);
  }
}
