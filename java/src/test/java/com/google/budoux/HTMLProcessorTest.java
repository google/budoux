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

import static org.junit.Assert.assertEquals;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HTMLProcessor}. */
@RunWith(JUnit4.class)
public class HTMLProcessorTest {
  String pre = "<span style=\"word-break: keep-all; overflow-wrap: anywhere;\">";
  String post = "</span>";

  private String wrap(String input) {
    return this.pre + input + this.post;
  }

  /** Creates a deterministic parser that breaks right before every given single-char target. */
  private static Parser breakBefore(String... targets) {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> uw4 = new HashMap<>();
    for (String t : targets) {
      uw4.put(t, 1000);
    }
    model.put("UW4", uw4);
    if (targets.length > 1) {
      Map<String, Integer> uw1 = new HashMap<>();
      uw1.put("\u0000", -(targets.length - 1) * 1000);
      model.put("UW1", uw1);
    }
    return new Parser(model);
  }

  /** Creates a deterministic parser that breaks right after every given single-char target. */
  private static Parser breakAfter(String... targets) {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> uw3 = new HashMap<>();
    for (String t : targets) {
      uw3.put(t, 1000);
    }
    model.put("UW3", uw3);
    if (targets.length > 1) {
      Map<String, Integer> uw1 = new HashMap<>();
      uw1.put("\u0000", -(targets.length - 1) * 1000);
      model.put("UW1", uw1);
    }
    return new Parser(model);
  }

  /** Creates a deterministic parser that breaks between every specified adjacent 2-char pair. */
  private static Parser breakBetweenPairs(String... pairs) {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> bw2 = new HashMap<>();
    for (String p : pairs) {
      bw2.put(p, 1000);
    }
    model.put("BW2", bw2);
    if (pairs.length > 1) {
      Map<String, Integer> uw1 = new HashMap<>();
      uw1.put("\u0000", -(pairs.length - 1) * 1000);
      model.put("UW1", uw1);
    }
    return new Parser(model);
  }

  /** Creates a deterministic parser that breaks at every character boundary. */
  private static Parser breakEverywhere() {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> uw4 = new HashMap<>();
    uw4.put("\u0000", -10000);
    model.put("UW4", uw4);
    return new Parser(model);
  }

  @Test
  public void testResolveWithSimpleTextInput() {
    Parser parser = breakBefore("d");
    String html = "abcdef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr>def"), result);
  }

  @Test
  public void testResolveWithStandardHTMLInput() {
    Parser parser = breakBefore("d");
    String html = "ab<a href=\"http://example.com\">cd</a>ef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("ab<a href=\"http://example.com\">c<wbr>d</a>ef"), result);
  }

  @Test
  public void testResolveWithImg() {
    Parser parser = breakBefore("d");
    String html = "<img>abcdef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("<img>abc<wbr>def"), result);
  }

  @Test
  public void testResolveWithUnpairedClose() {
    Parser parser = breakBefore("d");
    String html = "abcdef</p>";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr>def<p></p>"), result);
  }

  @Test
  public void testResolveWithNodesToSkip() {
    Parser parser = breakBefore("g");
    String html = "a<button>bcde</button>fghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("a<button>bcde</button>f<wbr>ghi"), result);
  }

  @Test
  public void testResolveWithSelectSkipNode() {
    Parser parser = breakBefore("g");
    String html = "<select><option>a</option><option>bcde</option></select>fghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(
        this.wrap("<select><option>a</option><option>bcde</option></select>f<wbr>ghi"), result);
  }

  @Test
  public void testResolveWithPreSkipNode() {
    Parser parser = breakBefore("g");
    String html = "<pre>abcde</pre>fghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("<pre>abcde</pre>f<wbr>ghi"), result);
  }

  @Test
  public void testResolveWithNodesBreakBeforeSkip() {
    Parser parser = breakBefore("d", "g", "j");
    String html = "abc<nobr>defghi</nobr>jkl";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr><nobr>defghi</nobr><wbr>jkl"), result);
  }

  @Test
  public void testResolveWithAfterSkip() {
    Parser parser = breakBefore("d", "g", "j");
    String html = "abc<nobr>def</nobr>ghijkl";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr><nobr>def</nobr><wbr>ghi<wbr>jkl"), result);
  }

  @Test
  public void testResolveWithAfterSkipWithImg() {
    Parser parser = breakBefore("d", "g", "j");
    String html = "abc<nobr>d<img>ef</nobr>ghijkl";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr><nobr>d<img>ef</nobr><wbr>ghi<wbr>jkl"), result);
  }

  @Test
  public void testResolveWithNothingToSplit() {
    Parser parser = new Parser(Collections.emptyMap());
    String html = "abcdef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abcdef"), result);
  }

  @Test
  public void testResolveBR() {
    Parser parser = breakEverywhere();
    String html = " 1  <br>  2 ";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap(" 1  <br>  2 "), result);
  }

  @Test
  public void testResolveSkipNodeAtTheEnd() {
    Parser parser = breakBefore("d", "g", "j");
    String html = "abcdefghijkl<img src=\"example.png\">";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr>def<wbr>ghi<wbr>jkl<img src=\"example.png\">"), result);
  }

  @Test
  public void testResolveWithComments() {
    Parser parser = breakBefore("d", "g", "j");
    String html = "abcdef<!-- comments should be ignored-->ghijkl";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(
        this.wrap("abc<wbr>def<!-- comments should be ignored--><wbr>ghi<wbr>jkl"), result);
  }

  @Test
  public void testResolveWithListItemsAndWhitespace() {
    Parser parser = breakBefore("d");
    String html = "<ul><li>abc</li>\n<li>def</li></ul>";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("<ul><li>abc</li>\n<li>def</li></ul>"), result);
  }

  @Test
  public void testResolveWithBreakAfterWhitespaceInList() {
    Parser parser = breakAfter("c");
    String html = "<ul><li>abc</li>\n<li>def</li></ul>";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("<ul><li>abc</li>\n<li>def</li></ul>"), result);
  }

  @Test
  public void testResolveEscapesText() {
    Parser parser = breakBefore(">");
    String html = "a&lt;b&gt;c&amp;d";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("a&lt;b<wbr>&gt;c&amp;d"), result);
  }

  @Test
  public void testResolveWithScript() {
    Parser parser = breakBefore("d");
    String html = "abc<script>if (a < b) f();</script>def";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<script>if (a < b) f();</script><wbr>def"), result);
  }

  @Test
  public void testResolveWithoutWrap() {
    Parser parser = breakBefore("d");
    String html = "ab<a href=\"http://example.com\">cd</a>ef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", false);
    assertEquals("ab<a href=\"http://example.com\">c<wbr>d</a>ef", result);
  }

  @Test
  public void testResolveWithExistingWbr() {
    Parser parser = breakBefore("d");
    String html = "abc<wbr>def";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<wbr>def"), result);
  }

  @Test
  public void testResolveWithExistingSeparators() {
    Parser parser = breakBefore("d", "g");
    String html = "abc<wbr>def\u200bghi";
    String result = parser.translateHTMLString(html);
    assertEquals(this.wrap("abc<wbr>def\u200bghi"), result);
  }

  @Test
  public void testResolveWithExistingSeparatorsAtStartOfChunk() {
    Parser parser = breakBefore("d");
    String html = "abc\u200bdef";
    String result = parser.translateHTMLString(html);
    assertEquals(this.wrap("abc\u200bdef"), result);
  }

  @Test
  public void testResolveWithStandaloneExistingSeparator() {
    Parser parser = breakBetweenPairs("cd");
    String html = "abc\u200bdef";
    String result = parser.translateHTMLString(html);
    assertEquals(this.wrap("abc\u200bdef"), result);
  }

  @Test
  public void testResolveWithExistingZwspBeforeSkipNode() {
    Parser parser = breakBefore("d");
    String html = "<span>abc\u200b</span><nobr>def</nobr>";
    String result = parser.translateHTMLString(html);
    assertEquals(this.wrap("<span>abc\u200b</span><nobr>def</nobr>"), result);
  }

  @Test
  public void testResolveWithExistingZwspAtStartOfSkipNode() {
    Parser parser = breakBefore("d");
    String html = "abc<nobr>\u200bdef</nobr>";
    String result = parser.translateHTMLString(html);
    assertEquals(this.wrap("abc<nobr>\u200bdef</nobr>"), result);
  }

  @Test
  public void testResolveWithBrPrecededByPhraseBoundary() {
    Parser parser = breakBefore("d", "e");
    String html = "abc<br>def";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<br>d<wbr>ef"), result);
  }

  @Test
  public void testResolveWithBrAtStartOfPhrase() {
    Parser parser = breakBefore("天");
    String html = "今日は<br>天気です。";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("今日は<br>天気です。"), result);
  }

  @Test
  public void testResolveWithTrailingBr() {
    Parser parser = breakAfter("c");
    String html = "abc<br>";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<br>"), result);
  }

  @Test
  public void testResolveWithBreakBeforeSkipPrecededByWhitespace() {
    Parser parser = breakBefore("d", "g");
    String html = "abc <button>def</button>ghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc <button>def</button>ghi"), result);
    assertEquals(
        this.wrap("abc <nobr>def</nobr><wbr>ghi"),
        HTMLProcessor.translate(parser, "abc <nobr>def</nobr>ghi", "<wbr>", true));
  }

  @Test
  public void testResolveWithBreakBeforeSkipStartingWithWhitespace() {
    Parser parser = breakBefore(" ", "d", "g");
    String html = "abc<button> def</button>ghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc<button> def</button><wbr>ghi"), result);
    assertEquals(
        this.wrap("abc<nobr> def</nobr><wbr>ghi"),
        HTMLProcessor.translate(parser, "abc<nobr> def</nobr>ghi", "<wbr>", true));
  }

  @Test
  public void testResolveWithBreakBeforeNoBreakSpace() {
    Parser parser = breakBefore("d");
    String html = "abc\u00a0def";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc&nbsp;def"), result);
  }

  @Test
  public void testResolveWithBreakAfterNoBreakSpace() {
    Parser parser = breakAfter("\u202f");
    String html = "abc\u202fdef";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc\u202fdef"), result);
  }

  @Test
  public void testResolveWithBreakBeforeSkipPrecededByNoBreakSpace() {
    Parser parser = breakBefore("d", "g");
    String html = "abc\u00a0<button>def</button>ghi";
    String result = HTMLProcessor.translate(parser, html, "<wbr>", true);
    assertEquals(this.wrap("abc&nbsp;<button>def</button>ghi"), result);
    assertEquals(
        this.wrap("abc&nbsp;<nobr>def</nobr><wbr>ghi"),
        HTMLProcessor.translate(parser, "abc\u00a0<nobr>def</nobr>ghi", "<wbr>", true));
  }

  @Test
  public void testBlockBoundariesIsolateParagraphs() {
    Parser breakBeforeA = breakBefore("a");
    assertEquals(
        "<p>xyz</p><p>abc</p>", breakBeforeA.translateHTMLString("<p>xyz</p><p>abc</p>", false));

    Map<String, Map<String, Integer>> lookaheadModel = new HashMap<>();
    Map<String, Integer> tw4 = new HashMap<>();
    tw4.put("bcd", 1000);
    lookaheadModel.put("TW4", tw4);
    Parser lookaheadParser = new Parser(lookaheadModel);
    assertEquals(
        "<p>ab</p><p>cd</p>", lookaheadParser.translateHTMLString("<p>ab</p><p>cd</p>", false));

    Parser pairParser = breakBetweenPairs("bc");
    assertEquals(
        "<div>ab<div>xy</div>cd</div>",
        pairParser.translateHTMLString("<div>ab<div>xy</div>cd</div>", false));
  }

  @Test
  public void testUnclosedNoBreakDoesNotLeakAcrossBlocks() {
    Parser parser = breakBefore("d");
    assertEquals(
        "<p><time>abcdef</time></p><p>abc\u200bdef</p>",
        parser.translateHTMLString("<p><time>abcdef</p><p>abcdef</p>", false));
  }

  @Test
  public void testSkipNodesExcludeContextFromSurroundingParagraph() {
    Parser parser = breakBetweenPairs("af");
    assertEquals(
        "a<button>bcde</button>\u200bfghi",
        parser.translateHTMLString("a<button>bcde</button>fghi", false));
    assertEquals(
        "a<select><option>bc</option><option>de</option></select>\u200bfghi",
        parser.translateHTMLString(
            "a<select><option>bc</option><option>de</option></select>fghi", false));
    assertEquals(
        "a<script>if (x < y && z > w) {}</script>\u200bfghi",
        parser.translateHTMLString("a<script>if (x < y && z > w) {}</script>fghi", false));
    assertEquals(
        "a<textarea>a&lt;\"bc&amp;d</textarea>\u200bfghi",
        parser.translateHTMLString("a<textarea>a&lt;\"bc&amp;d</textarea>fghi", false));
    assertEquals(
        "a<svg><text>bcde</text></svg>\u200bfghi",
        parser.translateHTMLString("a<svg><text>bcde</text></svg>fghi", false));
    assertEquals("a<pre>bcde</pre>fghi", parser.translateHTMLString("a<pre>bcde</pre>fghi", false));

    Parser preBoundaryParser = breakBetweenPairs("ef");
    assertEquals(
        "ae<pre>bcde</pre>fghi",
        preBoundaryParser.translateHTMLString("ae<pre>bcde</pre>fghi", false));
  }

  @Test
  public void testRubySkipsRtAndRpWithoutBreakingBaseContext() {
    Parser parser = breakBetweenPairs("bc", "xy");
    String html = "<ruby>ab<rp>(</rp><rt>xyz</rt><rp>)</rp></ruby>cd";
    assertEquals(
        "<ruby>ab<rp>(</rp><rt>xyz</rt><rp>)</rp></ruby>\u200bcd",
        parser.translateHTMLString(html, false));
  }

  @Test
  public void testNoBreakNodesProvideContextWithoutSplittingInside() {
    Parser parser = breakBetweenPairs("cd", "de", "fg");
    assertEquals(
        "abc\u200b<code>def</code>\u200bghi",
        parser.translateHTMLString("abc<code>def</code>ghi", false));
    assertEquals(
        "abc\u200b<time>def</time>\u200bghi",
        parser.translateHTMLString("abc<time>def</time>ghi", false));
    assertEquals(
        "abc\u200b<nobr><b>def</b></nobr>\u200bghi",
        parser.translateHTMLString("abc<nobr><b>def</b></nobr>ghi", false));

    Parser breakEvery = breakEverywhere();
    assertEquals(
        "a\u200bb\u200b<nobr>cd\u200b</nobr>e\u200bf",
        breakEvery.translateHTMLString("ab<nobr>cd\u200b</nobr>ef", false));
  }

  @Test
  public void testBrAndHrBreakParagraphsAndPreserveSurrogatePairs() {
    Parser breakEvery = breakEverywhere();
    assertEquals(
        "a\u200bb<br>𠮷\u200b野\u200b家", breakEvery.translateHTMLString("ab<br>𠮷野家", false));
    assertEquals(
        "<div>1<span>\u200b2\u200b3<br>4\u200b5</span>\u200b6</div>",
        breakEvery.translateHTMLString("<div>1<span>23<br>45</span>6</div>", false));
    assertEquals("a\u200bb<hr>c\u200bd", breakEvery.translateHTMLString("ab<hr>cd", false));
  }

  @Test
  public void testUffffAndCdataHandling() {
    Parser uffffParser = breakBefore("\uFFFF", "d");
    assertEquals(
        "abc\u200b\uFFFF\u200bdef", uffffParser.translateHTMLString("abc&#xffff;def", false));

    Parser parser = breakBefore("d");
    assertEquals(
        "<svg><![CDATA[abcdef]]></svg>abc\u200bdef",
        parser.translateHTMLString("<svg><![CDATA[abcdef]]></svg>abcdef", false));
    assertEquals("", parser.translateHTMLString("", false));
    assertEquals(this.wrap(""), parser.translateHTMLString("", true));
  }

  @Test
  public void testResolveWithPlainTextSeparatorContainingLessThan() {
    Parser parser = breakBefore("d");
    String html = "abcdef";
    String result = HTMLProcessor.translate(parser, html, " < ", false);
    assertEquals("abc &lt; def", result);
  }
}
