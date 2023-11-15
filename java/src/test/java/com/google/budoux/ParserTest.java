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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Parser}. */
@RunWith(JUnit4.class)
public class ParserTest {

  @Test
  public void testParse() {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> uw4 = new HashMap<>();
    uw4.put("a", 100);
    model.put("UW4", uw4);
    Parser parser = new Parser(model);
    List<String> result = parser.parse("xyzabc");
    List<String> expected = Arrays.asList("xyz", "abc");
    assertEquals(expected, result);
  }

  @Test
  public void testLoadDefaultJapaneseParser() {
    Parser parser = Parser.loadDefaultJapaneseParser();
    List<String> result = parser.parse("今日は天気です。");
    List<String> expected = Arrays.asList("今日は", "天気です。");
    assertEquals(expected, result);
  }

  @Test
  public void testTranslateHTMLString() {
    Map<String, Map<String, Integer>> model = new HashMap<>();
    Map<String, Integer> uw4 = new HashMap<>();
    uw4.put("a", 100);
    model.put("UW4", uw4);
    Parser parser = new Parser(model);
    String html = "<a href=\"http://example.com\">xyza</a>bc";
    String result = parser.translateHTMLString(html);
    assertEquals(
        "<span style=\"word-break: keep-all; overflow-wrap: anywhere;\"><a"
            + " href=\"http://example.com\">xyz\u200ba</a>bc</span>",
        result);
  }

  @Test
  public void testNewline() {
    Parser parser = Parser.loadDefaultJapaneseParser();
    List<String> result = parser.parse(" 1  \n  2 ");
    List<String> expected = Arrays.asList(" 1  \n  2 ");
    assertEquals(expected, result);
  }
}
