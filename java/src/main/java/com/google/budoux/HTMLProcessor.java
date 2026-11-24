/*
 * Copyright 2023 Google LLC
 *
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

package com.google.budoux;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.jsoup.Jsoup;
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;

/**
 * Processes HTML inputs into semantic line-broken HTML using Jsoup's DOM tree.
 *
 * <p>HTML elements are classified via {@link HtmlSpec} into block, inline, skip, no-break, forced
 * break, and break-opportunity actions, segmented per paragraph, and serialized back via Jsoup
 * without manual tag concatenation.
 */
final class HTMLProcessor {
  static final String STYLE = "word-break: keep-all; overflow-wrap: anywhere;";

  private static final char ZWSP = '\u200b';
  private static final Set<String> BLOCK_SKIP_NODES =
      Collections.unmodifiableSet(
          new HashSet<>(Arrays.asList("LISTING", "PLAINTEXT", "PRE", "XMP")));

  private enum DomAction {
    INLINE,
    BLOCK,
    SKIP,
    NO_BREAK,
    BREAK,
    BREAK_OPPORTUNITY
  }

  private HTMLProcessor() {}

  private static DomAction actionForElement(Element element) {
    String tag = element.normalName().toUpperCase(Locale.ENGLISH);
    HtmlSpec spec = HtmlSpec.INSTANCE;
    if (spec.skipNodes.contains(tag)) {
      return DomAction.SKIP;
    }
    if (spec.noBreakNodes.contains(tag)) {
      return DomAction.NO_BREAK;
    }
    if (spec.breakNodes.contains(tag)) {
      return DomAction.BREAK;
    }
    if (spec.breakOpportunityNodes.contains(tag)) {
      return DomAction.BREAK_OPPORTUNITY;
    }
    if (spec.blockNodes.contains(tag)) {
      return DomAction.BLOCK;
    }
    return DomAction.INLINE;
  }

  private static boolean isSpace(int codePoint) {
    return HtmlSpec.INSTANCE.spaceCodePoints.contains(codePoint);
  }

  private static boolean isAllSpaces(String text) {
    for (int offset = 0; offset < text.length(); ) {
      int cp = text.codePointAt(offset);
      if (!isSpace(cp)) {
        return false;
      }
      offset += Character.charCount(cp);
    }
    return true;
  }

  /**
   * Wraps either a mutable {@link TextNode} (`canSplit == true`) or an unbreakable context string
   * (`canSplit == false`, used for {@code NO_BREAK} elements such as {@code <nobr>} and {@code
   * <code>}).
   */
  private static final class NodeOrText {
    final TextNode textNode;
    final String unbreakableText;
    final List<String> chunks = new ArrayList<>();
    boolean hasBreakOpportunityAfter = false;

    NodeOrText(TextNode textNode) {
      this.textNode = textNode;
      this.unbreakableText = null;
    }

    NodeOrText(String unbreakableText) {
      this.textNode = null;
      this.unbreakableText = unbreakableText;
    }

    boolean canSplit() {
      return textNode != null;
    }

    String text() {
      return textNode != null ? textNode.getWholeText() : unbreakableText;
    }

    void split(String textSeparator, List<Node> nodeSeparatorPrototype) {
      if (!canSplit() || chunks.size() <= 1) {
        return;
      }
      if (nodeSeparatorPrototype == null) {
        textNode.text(String.join(textSeparator, chunks));
        return;
      }
      for (int i = 0; i < chunks.size(); i++) {
        String chunk = chunks.get(i);
        if (!chunk.isEmpty()) {
          textNode.before(new TextNode(chunk));
        }
        if (i < chunks.size() - 1) {
          for (Node sepNode : nodeSeparatorPrototype) {
            textNode.before(sepNode.clone());
          }
        }
      }
      textNode.remove();
    }
  }

  /** Represents an inline formatting unit (paragraph) collected for phrase segmentation. */
  private static final class Paragraph {
    final List<NodeOrText> nodes = new ArrayList<>();

    boolean isEmpty() {
      return nodes.isEmpty();
    }

    boolean hasSplittableNode() {
      for (NodeOrText node : nodes) {
        if (node.canSplit()) {
          return true;
        }
      }
      return false;
    }

    String text() {
      StringBuilder sb = new StringBuilder();
      for (NodeOrText node : nodes) {
        String t = node.text();
        if (t != null) {
          sb.append(t);
        }
      }
      return sb.toString();
    }

    void markBreakOpportunityAfterLastNode() {
      if (!nodes.isEmpty()) {
        nodes.get(nodes.size() - 1).hasBreakOpportunityAfter = true;
      }
    }

    Set<Integer> getForcedOpportunities() {
      Set<Integer> opportunities = new HashSet<>();
      int offset = 0;
      for (NodeOrText node : nodes) {
        String t = node.text();
        if (t != null) {
          for (int i = 0; i < t.length(); i++) {
            if (t.charAt(i) == ZWSP) {
              opportunities.add(offset + i);
              opportunities.add(offset + i + 1);
            }
          }
          offset += t.length();
        }
        if (node.hasBreakOpportunityAfter) {
          opportunities.add(offset);
        }
      }
      return opportunities;
    }
  }

  /** Mutable pointer to the active {@link Paragraph} within a block formatting context. */
  private static final class ParagraphContext {
    Paragraph current = new Paragraph();

    void flushInto(List<Paragraph> out) {
      if (!current.isEmpty()) {
        out.add(current);
        current = new Paragraph();
      }
    }
  }

  /**
   * Translates an HTML fragment string with semantic line-break separators and optional span
   * wrapping.
   *
   * @param parser the BudouX {@link Parser} instance.
   * @param html the input HTML string.
   * @param separator the separator string (plain text such as {@code "\u200b"} or HTML markup such
   *     as {@code "<wbr>"}).
   * @param wrap whether to wrap the output in a {@code <span>} element with inline styles.
   * @return the processed HTML string.
   */
  static String translate(Parser parser, String html, String separator, boolean wrap) {
    Document doc = Jsoup.parseBodyFragment(html);
    doc.outputSettings().prettyPrint(false).charset(StandardCharsets.UTF_8);

    Element body = doc.body();
    List<Paragraph> paragraphs = new ArrayList<>();
    collectFromElement(body, null, false, paragraphs);

    List<Node> nodeSeparatorPrototype = parseSeparatorIfElement(separator, body);
    for (Paragraph paragraph : paragraphs) {
      applyToParagraph(parser, paragraph, separator, nodeSeparatorPrototype);
    }

    String resolved = body.html();
    if (!wrap) {
      return resolved;
    }
    return String.format("<span style=\"%s\">%s</span>", STYLE, resolved);
  }

  private static List<Node> parseSeparatorIfElement(String separator, Element context) {
    if (separator.indexOf('<') < 0) {
      return null;
    }
    List<Node> parsed = org.jsoup.parser.Parser.parseFragment(separator, context, "");
    for (Node node : parsed) {
      if (node instanceof Element) {
        return parsed;
      }
    }
    return null;
  }

  private static void collectFromElement(
      Element element,
      ParagraphContext parentContext,
      boolean inNoBreak,
      List<Paragraph> outParagraphs) {
    DomAction action = actionForElement(element);
    if (action == DomAction.SKIP) {
      if (parentContext != null
          && BLOCK_SKIP_NODES.contains(element.normalName().toUpperCase(Locale.ENGLISH))) {
        parentContext.flushInto(outParagraphs);
      }
      return;
    }
    if (action == DomAction.BREAK) {
      if (parentContext != null && !parentContext.current.isEmpty()) {
        parentContext.flushInto(outParagraphs);
      }
      return;
    }
    if (action == DomAction.BREAK_OPPORTUNITY) {
      if (parentContext != null) {
        parentContext.current.markBreakOpportunityAfterLastNode();
      }
      return;
    }

    boolean isNewBlock = parentContext == null || action == DomAction.BLOCK;
    if (isNewBlock && parentContext != null) {
      parentContext.flushInto(outParagraphs);
    }

    ParagraphContext ctx = isNewBlock ? new ParagraphContext() : parentContext;
    boolean childNoBreak = inNoBreak || action == DomAction.NO_BREAK;

    for (Node child : element.childNodes()) {
      collectFromChildNode(child, ctx, childNoBreak, outParagraphs);
    }

    if (isNewBlock) {
      ctx.flushInto(outParagraphs);
    }
  }

  private static void collectFromChildNode(
      Node child, ParagraphContext ctx, boolean inNoBreak, List<Paragraph> outParagraphs) {
    if (child instanceof Element) {
      collectFromElement((Element) child, ctx, inNoBreak, outParagraphs);
    } else if (child instanceof TextNode && !(child instanceof CDataNode)) {
      TextNode textNode = (TextNode) child;
      String wholeText = textNode.getWholeText();
      if (wholeText.isEmpty()) {
        return;
      }
      if (inNoBreak) {
        ctx.current.nodes.add(new NodeOrText(wholeText));
      } else {
        ctx.current.nodes.add(new NodeOrText(textNode));
      }
    }
  }

  private static void applyToParagraph(
      Parser parser, Paragraph paragraph, String separator, List<Node> nodeSeparatorPrototype) {
    if (!paragraph.hasSplittableNode()) {
      return;
    }
    String text = paragraph.text();
    if (text.isEmpty() || isAllSpaces(text)) {
      return;
    }

    int[] boundaries = parser.parseBoundaries(text);
    if (boundaries.length == 0) {
      return;
    }
    Set<Integer> forcedOpportunities = paragraph.getForcedOpportunities();

    List<Integer> adjustedBoundaries = new ArrayList<>(boundaries.length + 1);
    for (int boundary : boundaries) {
      if (boundary <= 0 || boundary >= text.length()) {
        continue;
      }
      if (forcedOpportunities.contains(boundary)) {
        continue;
      }
      int cpBefore = Character.codePointBefore(text, boundary);
      int cpAt = Character.codePointAt(text, boundary);
      if (isSpace(cpBefore) || isSpace(cpAt)) {
        continue;
      }
      adjustedBoundaries.add(boundary);
    }

    if (!adjustedBoundaries.isEmpty()) {
      adjustedBoundaries.add(text.length() + 1);
      splitNodes(paragraph.nodes, adjustedBoundaries, separator, nodeSeparatorPrototype);
    }
  }

  private static void splitNodes(
      List<NodeOrText> nodes,
      List<Integer> boundaries,
      String textSeparator,
      List<Node> nodeSeparatorPrototype) {
    int boundaryIndex = 0;
    int boundary = boundaries.get(0);
    int nodeStart = 0;
    NodeOrText lastSplittableNode = null;

    for (NodeOrText node : nodes) {
      String nodeText = node.text();
      if (nodeText == null || nodeText.isEmpty()) {
        continue;
      }
      int nodeLength = nodeText.length();
      int nodeEnd = nodeStart + nodeLength;

      if (!node.canSplit()) {
        if (lastSplittableNode != null && boundary == nodeStart) {
          if (lastSplittableNode.chunks.isEmpty()) {
            lastSplittableNode.chunks.add(lastSplittableNode.text());
          }
          lastSplittableNode.chunks.add("");
        }
        while (boundary < nodeEnd) {
          boundary = boundaries.get(++boundaryIndex);
        }
        lastSplittableNode = null;
        nodeStart = nodeEnd;
        continue;
      }

      lastSplittableNode = node;
      if (boundary >= nodeEnd) {
        nodeStart = nodeEnd;
        continue;
      }

      int chunkStartInNode = 0;
      while (boundary < nodeEnd) {
        int boundaryInNode = boundary - nodeStart;
        node.chunks.add(nodeText.substring(chunkStartInNode, boundaryInNode));
        chunkStartInNode = boundaryInNode;
        boundary = boundaries.get(++boundaryIndex);
      }
      node.chunks.add(nodeText.substring(chunkStartInNode));
      nodeStart = nodeEnd;
    }

    for (NodeOrText node : nodes) {
      node.split(textSeparator, nodeSeparatorPrototype);
    }
  }
}
