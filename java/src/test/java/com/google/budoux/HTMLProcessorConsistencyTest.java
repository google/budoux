/*
 * Copyright 2024 Google LLC
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

import com.google.gson.Gson;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Consistency tests for {@link HTMLProcessor}. */
@RunWith(JUnit4.class)
public class HTMLProcessorConsistencyTest {

  static class TestCase {
    String description;
    String html;
    String expected;
  }

  @Test
  public void testSharedCases() throws IOException {
    Gson gson = new Gson();
    String jsonPath = "../tests/html_processor_shared_results.json";
    Reader reader = Files.newBufferedReader(Paths.get(jsonPath), StandardCharsets.UTF_8);
    TestCase[] testCases = gson.fromJson(reader, TestCase[].class);

    // We use the default Japanese parser for these tests.
    // For simplicity, we assume the model is loaded correctly.
    Parser parser = Parser.loadDefaultJapaneseParser();

    for (TestCase tc : testCases) {
      String text = HTMLProcessor.getText(tc.html);
      List<String> chunks = parser.parse(text);
      String result = HTMLProcessor.resolve(chunks, tc.html, "\u200b");
      String expectedWrapped = "<span style=\"word-break: keep-all; overflow-wrap: anywhere;\">" + tc.expected + "</span>";
      assertEquals("Failed consistency case: " + tc.description, expectedWrapped, result);
    }
  }
}
