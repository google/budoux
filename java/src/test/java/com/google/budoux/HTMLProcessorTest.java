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

/** Unit tests for {@link HTMLProcessor}. */
public class HTMLProcessorTest {

  @Test
  public void testResolveWithSimpleTextInput() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "abcdef";
    String result = HTMLProcessor.resolve(phrases, html);
    assertEquals(
        "<span style=\"word-break: keep-all; overflow-wrap: break-word;\">abc<wbr>def</span>",
        result);
  }

  @Test
  public void testResolveWithStandardHTMLInput() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "ab<a href=\"http://example.com\">cd</a>ef";
    String result = HTMLProcessor.resolve(phrases, html);
    assertEquals(
        "<span style=\"word-break: keep-all; overflow-wrap: break-word;\">ab<a"
            + " href=\"http://example.com\">c<wbr>d</a>ef</span>",
        result);
  }

  @Test
  public void testResolveWithNodesToSkip() {
    List<String> phrases = Arrays.asList("abc", "def");
    String html = "a<button>bcde</button>f";
    String result = HTMLProcessor.resolve(phrases, html);
    assertEquals(
        "<span style=\"word-break: keep-all; overflow-wrap:"
            + " break-word;\">a<button>bcde</button>f</span>",
        result);
  }

  @Test
  public void testResolveWithNothingToSplit() {
    List<String> phrases = Arrays.asList("abcdef");
    String html = "abcdef";
    String result = HTMLProcessor.resolve(phrases, html);
    assertEquals(
        "<span style=\"word-break: keep-all; overflow-wrap: break-word;\">abcdef</span>", result);
  }

  @Test
  public void testGetText() {
    String html = "Hello <button><b>W</b>orld</button>!";
    String result = HTMLProcessor.getText(html);
    assertEquals("Hello World!", result);
  }
}
