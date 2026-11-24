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

import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The BudouX parser that translates the input sentence into phrases.
 *
 * <p>You can create a parser instance by invoking {@code new Parser(model)} with the model data you
 * want to use. You can also create a parser by specifying the model file path with {@code
 * Parser.loadByFileName(modelFileName)}.
 *
 * <p>In most cases, it's sufficient to use the default parser for the language. For example, you
 * can create a default Japanese parser as follows.
 *
 * <pre>
 * Parser parser = Parser.loadDefaultJapaneseParser();
 * </pre>
 */
public class Parser {
  /** Recommended CSS style declarations for parent elements containing BudouX-segmented text. */
  public static final String PARENT_CSS_STYLE = HTMLProcessor.STYLE;

  private static final Set<Integer> invisibleFormatCodePoints =
      HtmlSpec.INSTANCE.invisibleFormatCodePoints;

  private final long totalScore;
  private final Map<String, Integer> uw1;
  private final Map<String, Integer> uw2;
  private final Map<String, Integer> uw3;
  private final Map<String, Integer> uw4;
  private final Map<String, Integer> uw5;
  private final Map<String, Integer> uw6;
  private final Map<String, Integer> bw1;
  private final Map<String, Integer> bw2;
  private final Map<String, Integer> bw3;
  private final Map<String, Integer> tw1;
  private final Map<String, Integer> tw2;
  private final Map<String, Integer> tw3;
  private final Map<String, Integer> tw4;

  /**
   * Constructs a BudouX parser.
   *
   * @param model the model data.
   */
  public Parser(Map<String, Map<String, Integer>> model) {
    long sum = 0;
    for (Map<String, Integer> group : model.values()) {
      for (int weight : group.values()) {
        sum += weight;
      }
    }
    this.totalScore = sum;
    this.uw1 = copyGroup(model, "UW1");
    this.uw2 = copyGroup(model, "UW2");
    this.uw3 = copyGroup(model, "UW3");
    this.uw4 = copyGroup(model, "UW4");
    this.uw5 = copyGroup(model, "UW5");
    this.uw6 = copyGroup(model, "UW6");
    this.bw1 = copyGroup(model, "BW1");
    this.bw2 = copyGroup(model, "BW2");
    this.bw3 = copyGroup(model, "BW3");
    this.tw1 = copyGroup(model, "TW1");
    this.tw2 = copyGroup(model, "TW2");
    this.tw3 = copyGroup(model, "TW3");
    this.tw4 = copyGroup(model, "TW4");
  }

  private static Map<String, Integer> copyGroup(
      Map<String, Map<String, Integer>> model, String key) {
    Map<String, Integer> group = model.get(key);
    return group != null ? new HashMap<>(group) : Collections.emptyMap();
  }

  /**
   * Loads the default Japanese parser.
   *
   * @return a BudouX parser with the default Japanese model.
   */
  public static Parser loadDefaultJapaneseParser() {
    return loadByFileName("/models/ja.json");
  }

  /**
   * Loads the default Simplified Chinese parser.
   *
   * @return a BudouX parser with the default Simplified Chinese model.
   */
  public static Parser loadDefaultSimplifiedChineseParser() {
    return loadByFileName("/models/zh-hans.json");
  }

  /**
   * Loads the default Traditional Chinese parser.
   *
   * @return a BudouX parser with the default Traditional Chinese model.
   */
  public static Parser loadDefaultTraditionalChineseParser() {
    return loadByFileName("/models/zh-hant.json");
  }

  /**
   * Loads the default Thai parser.
   *
   * @return a BudouX parser with the default Thai model.
   */
  public static Parser loadDefaultThaiParser() {
    return loadByFileName("/models/th.json");
  }

  /**
   * Loads a parser by specifying the model file path.
   *
   * @param modelFileName the model file path.
   * @return a BudouX parser.
   * @throws IllegalArgumentException if the model resource does not exist.
   */
  public static Parser loadByFileName(String modelFileName) {
    Gson gson = new Gson();
    Type type = new TypeToken<Map<String, Map<String, Integer>>>() {}.getType();
    InputStream inputStream = Parser.class.getResourceAsStream(modelFileName);
    if (inputStream == null) {
      throw new IllegalArgumentException("Model resource not found: " + modelFileName);
    }
    try (Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
      Map<String, Map<String, Integer>> model = gson.fromJson(reader, type);
      return new Parser(model);
    } catch (JsonIOException | JsonSyntaxException | IOException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Parses the input sentence and returns UTF-16 code unit offsets of phrase boundaries.
   *
   * <p>Invisible format characters (such as U+200B ZERO WIDTH SPACE) are treated as transparent in
   * the feature window so repeated segmentation stays idempotent.
   *
   * @param sentence the sentence to analyze.
   * @return an array of UTF-16 code unit boundary offsets.
   */
  public int[] parseBoundaries(String sentence) {
    if (sentence.isEmpty()) {
      return new int[0];
    }
    int maxCodePoints = sentence.codePointCount(0, sentence.length());
    String[] chars = new String[maxCodePoints];
    int[] offsets = new int[maxCodePoints];

    int visibleCount = 0;
    for (int charOffset = 0; charOffset < sentence.length(); ) {
      int codePoint = sentence.codePointAt(charOffset);
      int charCount = Character.charCount(codePoint);
      if (invisibleFormatCodePoints.contains(codePoint)) {
        charOffset += charCount;
        continue;
      }
      chars[visibleCount] = sentence.substring(charOffset, charOffset + charCount);
      offsets[visibleCount] = charOffset;
      visibleCount++;
      charOffset += charCount;
    }

    if (visibleCount <= 1) {
      return new int[0];
    }

    int[] tempBoundaries = new int[visibleCount - 1];
    int boundaryCount = 0;
    for (int i = 1; i < visibleCount; i++) {
      long score = -this.totalScore;
      if (i > 2) {
        if (!this.uw1.isEmpty()) {
          score += 2L * this.uw1.getOrDefault(chars[i - 3], 0);
        }
        if (!this.tw1.isEmpty()) {
          score += 2L * this.tw1.getOrDefault(chars[i - 3] + chars[i - 2] + chars[i - 1], 0);
        }
      }
      if (i > 1) {
        if (!this.uw2.isEmpty()) {
          score += 2L * this.uw2.getOrDefault(chars[i - 2], 0);
        }
        if (!this.bw1.isEmpty()) {
          score += 2L * this.bw1.getOrDefault(chars[i - 2] + chars[i - 1], 0);
        }
        if (!this.tw2.isEmpty()) {
          score += 2L * this.tw2.getOrDefault(chars[i - 2] + chars[i - 1] + chars[i], 0);
        }
      }
      if (!this.uw3.isEmpty()) {
        score += 2L * this.uw3.getOrDefault(chars[i - 1], 0);
      }
      if (!this.uw4.isEmpty()) {
        score += 2L * this.uw4.getOrDefault(chars[i], 0);
      }
      if (!this.bw2.isEmpty()) {
        score += 2L * this.bw2.getOrDefault(chars[i - 1] + chars[i], 0);
      }
      if (i + 1 < visibleCount) {
        if (!this.uw5.isEmpty()) {
          score += 2L * this.uw5.getOrDefault(chars[i + 1], 0);
        }
        if (!this.bw3.isEmpty()) {
          score += 2L * this.bw3.getOrDefault(chars[i] + chars[i + 1], 0);
        }
        if (!this.tw3.isEmpty()) {
          score += 2L * this.tw3.getOrDefault(chars[i - 1] + chars[i] + chars[i + 1], 0);
        }
      }
      if (i + 2 < visibleCount) {
        if (!this.uw6.isEmpty()) {
          score += 2L * this.uw6.getOrDefault(chars[i + 2], 0);
        }
        if (!this.tw4.isEmpty()) {
          score += 2L * this.tw4.getOrDefault(chars[i] + chars[i + 1] + chars[i + 2], 0);
        }
      }
      if (score > 0) {
        tempBoundaries[boundaryCount++] = offsets[i];
      }
    }
    return Arrays.copyOf(tempBoundaries, boundaryCount);
  }

  /**
   * Parses a sentence into phrases.
   *
   * @param sentence the sentence to break by phrase.
   * @return a list of phrases.
   */
  public List<String> parse(String sentence) {
    if (sentence.isEmpty()) {
      return new ArrayList<>();
    }
    int[] boundaries = parseBoundaries(sentence);
    List<String> result = new ArrayList<>(boundaries.length + 1);
    int start = 0;
    for (int boundary : boundaries) {
      result.add(sentence.substring(start, boundary));
      start = boundary;
    }
    result.add(sentence.substring(start));
    return result;
  }

  /**
   * Translates an HTML string with phrases wrapped in no-breaking markup.
   *
   * @param html an HTML string.
   * @return the translated HTML string with no-breaking markup.
   */
  public String translateHTMLString(String html) {
    return translateHTMLString(html, true);
  }

  /**
   * Translates an HTML string with phrases separated for semantic line breaks.
   *
   * @param html an HTML string.
   * @param wrap whether to wrap the output in a span element with inline styles.
   * @return the translated HTML string.
   */
  public String translateHTMLString(String html, boolean wrap) {
    return HTMLProcessor.translate(this, html, "\u200b", wrap);
  }
}
