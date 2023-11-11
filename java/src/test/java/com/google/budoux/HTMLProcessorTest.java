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

import java.util.Arrays;
import java.util.List;
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

  @Test
  public void testResolveWithSimpleTextInput() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "abcdef";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("abc<wbr>def"), result);
  }

  @Test
  public void testResolveWithStandardHTMLInput() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "ab<a href=\"http://example.com\">cd</a>ef";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(
        this.wrap("ab<a href=\"http://example.com\">c<wbr>d</a>ef"),
        result);
  }

  @Test
  public void testResolveWithImg() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "<img>abcdef";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("<img>abc<wbr>def"), result);
  }

  @Test
  public void testResolveWithUnpairedClose() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "abcdef</p>";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("abc<wbr>def<p></p>"), result);
  }

  @Test
  public void testResolveWithNodesToSkip() {
    List<String> phrases = Arrays.asList("abc", "def", "ghi");
    String html = "a<button>bcde</button>fghi";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("a<button>bcde</button>f<wbr>ghi"), result);
  }

  @Test
  public void testResolveWithNodesBreakBeforeSkip() {
    List<String> phrases = Arrays.asList("abc", "def", "ghi", "jkl");
    String html = "abc<nobr>defghi</nobr>jkl";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("abc<wbr><nobr>defghi</nobr><wbr>jkl"), result);
  }

  @Test
  public void testResolveWithAfterSkip() {
    List<String> phrases = Arrays.asList("abc", "def", "ghi", "jkl");
    String html = "abc<nobr>def</nobr>ghijkl";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(
        this.wrap("abc<wbr><nobr>def</nobr><wbr>ghi<wbr>jkl"),
        result);
  }

  @Test
  public void testResolveWithAfterSkipWithImg() {
    List<String> phrases = Arrays.asList("abc", "def", "ghi", "jkl");
    String html = "abc<nobr>d<img>ef</nobr>ghijkl";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(
        this.wrap("abc<wbr><nobr>d<img>ef</nobr><wbr>ghi<wbr>jkl"),
        result);
  }

  @Test
  public void testResolveWithNothingToSplit() {
    List<String> phrases = Arrays.asList("abcdef");
    String html = "abcdef";
    String result = HTMLProcessor.resolve(phrases, html, "<wbr>");
    assertEquals(this.wrap("abcdef"), result);
  }

  @Test
  public void testGetText() {
    String html = "Hello <button><b>W</b>orld</button>!";
    String result = HTMLProcessor.getText(html);
    assertEquals("Hello World!", result);
  }
}
