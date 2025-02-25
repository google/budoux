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

import {parseFromString} from './dom.js';
import {Parser} from './parser.js';

const assert = console.assert;

const ZWSP_CODEPOINT = 0x200b; // U+200B ZERO WIDTH SPACE
const ZWSP = String.fromCharCode(ZWSP_CODEPOINT);

// We could use `Node.TEXT_NODE` and `Node.ELEMENT_NODE` in a browser context,
// but we define the same here for Node.js environments.
const NodeType = {
  ELEMENT_NODE: 1,
  TEXT_NODE: 3,
};

const DomAction = {
  Inline: 0, // An inline content, becomes a part of a paragraph.
  Block: 1, // A nested paragraph.
  Skip: 2, // Skip the content. The content before and after are connected.
  Break: 3, // A forced break. The content before and after become paragraphs.
  NoBreak: 4, // The content provides context, but it's not breakable.
  BreakOpportunity: 5, // Force a break opportunity.
} as const;
type DomAction = (typeof DomAction)[keyof typeof DomAction];

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
  WBR: DomAction.BreakOpportunity,

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

  // Deprecated, but supported in all browsers.
  // https://developer.mozilla.org/en-US/docs/Web/HTML/Element/nobr
  NOBR: DomAction.NoBreak,
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
 * Determine the action for a CSS `display` property value.
 * @param display The value of the CSS `display` property.
 * @return The {@link domActions} for the value.
 */
function actionForDisplay(display: string): DomAction {
  // Handle common cases first.
  if (display === 'inline') return DomAction.Inline;
  if (display === 'block') return DomAction.Block;

  // Handle Ruby base as in-flow.
  if (display.startsWith('ruby')) {
    if (display === 'ruby-text') return DomAction.Skip;
    return DomAction.Inline;
  }

  // Handle other values including multi-value syntax as blocks.
  // https://drafts.csswg.org/css-display/#the-display-properties
  return DomAction.Block;
}

/**
 * Determine the action for an element.
 * @param element An element to determine the action for.
 * @return The {@link domActions} for the element.
 */
function actionForElement(element: Element): DomAction {
  const nodeName = element.nodeName;
  const action = domActions[nodeName];
  if (action !== undefined) return action;

  if (typeof globalThis.getComputedStyle === 'function') {
    const style = globalThis.getComputedStyle(element);
    switch (style.whiteSpace) {
      case 'nowrap':
      case 'pre':
        return DomAction.NoBreak;
    }

    const display = style.display;
    if (display) return actionForDisplay(display);
    // `display` is an empty string if the element is not connected.
  }

  // Use the built-in rules if the `display` property is empty, or if
  // `getComputedStyle` is missing (e.g., jsdom.)
  return defaultBlockElements.has(nodeName)
    ? DomAction.Block
    : DomAction.Inline;
}

/**
 * Applies wrapping styles to make linebreak controls work in children.
 * @param element A parent element to apply the styles.
 */
export const applyWrapStyle = (element: HTMLElement) => {
  element.style.wordBreak = 'keep-all';
  element.style.overflowWrap = 'anywhere';
};

/**
 * Represents a node in {@link Paragraph}.
 *
 * It wraps a {@link Text} or a {@link string}.
 *
 * A {@link string} provides the context for the parser, but it can't be split.
 */
class NodeOrText {
  nodeOrText: Text | string;
  chunks: string[] = [];
  hasBreakOpportunityAfter = false;

  constructor(nodeOrText: Text | string) {
    this.nodeOrText = nodeOrText;
  }

  private static isString(value: Text | string): value is string {
    return typeof value === 'string';
  }

  get canSplit(): boolean {
    return !NodeOrText.isString(this.nodeOrText);
  }

  get text(): string | null {
    return NodeOrText.isString(this.nodeOrText)
      ? this.nodeOrText
      : this.nodeOrText.nodeValue;
  }

  get length(): number {
    return this.text?.length ?? 0;
  }

  /**
   * Split the {@link Text} in the same way as the {@link chunks}.
   * Joining all {@link chunks} must be equal to {@link text}.
   */
  split(separator: string | Node) {
    const chunks = this.chunks;
    assert(chunks.length === 0 || chunks.join('') === this.text);
    if (chunks.length <= 1) return;
    if (NodeOrText.isString(this.nodeOrText)) return;
    const node = this.nodeOrText;
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
      // Add a separator between chunks. To simplify the logic, add a separator
      // after each chunk, then remove the last one.
      // To avoid `cloneNode` for the temporary one that is going to be removed,
      // add `null` as a marker, then replace them with `cloneNode` later.
      nodes.push(null);
    }
    nodes.pop();
    nodes = nodes.map(n => (n ? n : separator.cloneNode(true)));
    node.replaceWith(...nodes);
  }
}
export class NodeOrTextForTesting extends NodeOrText {}

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
  nodes: NodeOrText[] = [];

  constructor(element: HTMLElement) {
    this.element = element;
  }

  isEmpty(): boolean {
    return this.nodes.length === 0;
  }
  get text(): string {
    return this.nodes.map(node => node.text).join('');
  }

  get lastNode(): NodeOrText | undefined {
    return this.nodes.length ? this.nodes[this.nodes.length - 1] : undefined;
  }
  setHasBreakOpportunityAfter() {
    const lastNode = this.lastNode;
    if (lastNode) lastNode.hasBreakOpportunityAfter = true;
  }

  /**
   * @return Indices of forced break opportunities in the source.
   * They can be created by `<wbr>` tag or `&ZeroWidthSpace;`.
   */
  getForcedOpportunities(): number[] {
    const opportunities: number[] = [];
    let len = 0;
    for (const node of this.nodes) {
      if (node.canSplit) {
        const text = node.text;
        if (text) {
          for (let i = 0; i < text.length; ++i) {
            if (text.charCodeAt(i) === ZWSP_CODEPOINT) {
              opportunities.push(len + i + 1);
            }
          }
        }
      }
      len += node.length;
      if (node.hasBreakOpportunityAfter) {
        opportunities.push(len);
      }
    }
    return opportunities;
  }

  /**
   * @return Filtered {@param boundaries} by excluding
   * {@link getForcedOpportunities} if it's not empty.
   * Otherwise {@param boundaries}.
   */
  excludeForcedOpportunities(boundaries: number[]): number[] {
    const forcedOpportunities = this.getForcedOpportunities();
    if (!forcedOpportunities.length) return boundaries;
    const set = new Set<number>(forcedOpportunities);
    return boundaries.filter(i => !set.has(i));
  }
}
export class ParagraphForTesting extends Paragraph {}

/**
 * Options for {@link HTMLProcessor}.
 */
export interface HTMLProcessorOptions {
  /**
   * This class name is added to the containing block when the BudouX is applied.
   * The containing block should have following CSS properties to make it work.
   * `{ word-break: keep-all; overflow-wrap: anywhere; }`
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
 * Adds HTML processing support to a BudouX {@link Parser}.
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
   * Checks if the given element has a text node in its children.
   *
   * @param ele An element to be checked.
   * @return Whether the element has a child text node.
   */
  static hasChildTextNode(ele: HTMLElement) {
    for (const child of ele.childNodes) {
      if (child.nodeType === NodeType.TEXT_NODE) return true;
    }
    return false;
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
      assert(!block.isEmpty());
      this.applyToParagraph(block);
    }
  }

  /**
   * Find paragraphs from a given HTML element.
   * @param element The root element to find paragraphs.
   * @param parent The parent {@link Paragraph} if any.
   * @return A list of {@link Paragraph}s.
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
      if (parent && !parent.isEmpty()) {
        parent.setHasBreakOpportunityAfter();
        yield parent;
        parent.nodes = [];
      }
      assert(!element.firstChild);
      return;
    }
    if (action === DomAction.BreakOpportunity) {
      if (parent) parent.setHasBreakOpportunityAfter();
      return;
    }

    // Determine if this element creates a new inline formatting context, or if
    // this element belongs to the parent inline formatting context.
    assert(
      action === DomAction.Block ||
        action === DomAction.Inline ||
        action === DomAction.NoBreak
    );
    const isNewBlock = !parent || action === DomAction.Block;
    const block = isNewBlock ? new Paragraph(element) : parent;

    // Collect all text nodes in this inline formatting context, while searching
    // descendant elements recursively.
    for (const child of element.childNodes) {
      switch (child.nodeType) {
        case NodeType.ELEMENT_NODE:
          for (const childBlock of this.getBlocks(child as HTMLElement, block))
            yield childBlock;
          break;
        case NodeType.TEXT_NODE:
          if (action === DomAction.NoBreak) {
            const text = child.nodeValue;
            if (text) {
              block.nodes.push(new NodeOrText(text));
            }
            break;
          }
          block.nodes.push(new NodeOrText(child as Text));
          break;
      }
    }

    // Apply if this is an inline formatting context.
    if (isNewBlock && !block.isEmpty()) yield block;
  }

  /**
   * Apply the BudouX to the given {@link Paragraph}.
   * @param paragraph The {@link Paragraph} to apply.
   */
  applyToParagraph(paragraph: Paragraph): void {
    assert(paragraph.nodes.length > 0);
    if (!paragraph.nodes.some(node => node.canSplit)) return;
    const text = paragraph.text;
    // No changes if whitespace-only.
    if (/^\s*$/.test(text)) return;

    // Compute the phrase boundaries.
    const boundaries = this.parser_.parseBoundaries(text);
    // No changes if single phrase.
    if (boundaries.length <= 0) return;
    // The boundaries should be between 1 and `text.length - 1` in the
    // ascending order.
    assert(boundaries[0] > 0);
    assert(boundaries.every((x, i) => i === 0 || x > boundaries[i - 1]));
    assert(boundaries[boundaries.length - 1] < text.length);

    const adjustedBoundaries = paragraph.excludeForcedOpportunities(boundaries);

    // Add a sentinel to help iterating.
    adjustedBoundaries.push(text.length + 1);

    this.splitNodes(paragraph.nodes, adjustedBoundaries);
    this.applyBlockStyle(paragraph.element);
  }

  /**
   * Split {@link NodeOrText} at the specified boundaries.
   * @param nodes A list of {@link NodeOrText}.
   * @param boundaries A list of indices of the text to split at.
   */
  splitNodes(nodes: NodeOrText[], boundaries: number[]): void {
    assert(boundaries.length > 0);
    assert(boundaries.every((x, i) => i === 0 || x > boundaries[i - 1]));
    const textLen = nodes.reduce((sum, node) => sum + node.length, 0);
    // The last boundary must be a sentinel.
    assert(boundaries[boundaries.length - 1] > textLen);

    // Distribute `boundaries` to `node.chunks`.
    let boundary_index = 0;
    let boundary = boundaries[0];
    assert(boundary > 0);
    let nodeStart = 0; // the start index of the `nodeText` in the whole text.
    let lastNode: NodeOrText | null = null;
    for (const node of nodes) {
      assert(boundary >= nodeStart);
      assert(node.chunks.length === 0);
      const nodeText = node.text;
      if (!nodeText) continue;
      const nodeLength = nodeText.length;
      const nodeEnd = nodeStart + nodeLength;
      assert(!lastNode || lastNode.canSplit);
      if (!node.canSplit) {
        // If there's a boundary between nodes and `lastNode.canSplit`, add a
        // boundary to the end of the `lastNode`.
        if (lastNode && boundary === nodeStart) {
          if (lastNode.chunks.length === 0)
            lastNode.chunks.push(lastNode.text ?? '');
          lastNode.chunks.push('');
        }
        while (boundary < nodeEnd) {
          boundary = boundaries[++boundary_index];
        }
        lastNode = null;
        nodeStart = nodeEnd;
        continue;
      }

      // Check if the next boundary is in this `node`.
      lastNode = node;
      if (boundary >= nodeEnd) {
        nodeStart = nodeEnd;
        continue;
      }

      // Compute the boundary indices in the `node`.
      const chunks = node.chunks;
      let chunkStartInNode = 0;
      while (boundary < nodeEnd) {
        const boundaryInNode = boundary - nodeStart;
        assert(boundaryInNode >= chunkStartInNode);
        chunks.push(nodeText.slice(chunkStartInNode, boundaryInNode));
        chunkStartInNode = boundaryInNode;
        boundary = boundaries[++boundary_index];
      }
      // Add the rest of the `nodeText`.
      assert(chunkStartInNode < nodeLength);
      chunks.push(nodeText.slice(chunkStartInNode));

      nodeStart = nodeEnd;
    }
    // Check if all nodes and boundaries are consumed.
    assert(nodeStart === textLen);
    assert(boundary_index < boundaries.length);
    assert(boundaries[boundary_index] >= textLen);

    // `node.chunks` are finalized. Split them.
    for (const node of nodes) {
      node.split(this.separator);
    }
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
    applyWrapStyle(element);
  }
}

/**
 * BudouX {@link Parser} with HTML processing support.
 */
export class HTMLProcessingParser extends Parser {
  htmlProcessor: HTMLProcessor;

  constructor(
    model: {[key: string]: {[key: string]: number}},
    htmlProcessorOptions: HTMLProcessorOptions = {
      separator: ZWSP,
    }
  ) {
    super(model);
    this.htmlProcessor = new HTMLProcessor(this, htmlProcessorOptions);
  }

  /**
   * Applies markups for semantic line breaks to the given HTML element.
   * @param parentElement The input element.
   */
  applyToElement(parentElement: HTMLElement) {
    this.htmlProcessor.applyToElement(parentElement);
  }

  /**
   * Translates the given HTML string to another HTML string with markups
   * for semantic line breaks.
   * @param html An input html string.
   * @return The translated HTML string.
   */
  translateHTMLString(html: string) {
    if (html === '') return html;
    const doc = parseFromString(html);
    if (HTMLProcessor.hasChildTextNode(doc.body)) {
      const wrapper = doc.createElement('span') as unknown as HTMLElement;
      wrapper.append(...doc.body.childNodes);
      doc.body.append(wrapper);
    }
    this.applyToElement(doc.body.childNodes[0] as HTMLElement);
    return doc.body.innerHTML;
  }
}
