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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
  private final int totalScore;
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
    int sum = 0;
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
   */
  public static Parser loadByFileName(String modelFileName) {
    Gson gson = new Gson();
    Type type = new TypeToken<Map<String, Map<String, Integer>>>() {}.getType();
    InputStream inputStream = Parser.class.getResourceAsStream(modelFileName);
    try (Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
      Map<String, Map<String, Integer>> model = gson.fromJson(reader, type);
      return new Parser(model);
    } catch (JsonIOException | JsonSyntaxException | IOException e) {
      throw new AssertionError(e);
    }
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
    List<String> result = new ArrayList<>();
    int phraseStart = 0;
    int length = sentence.length();
    for (int i = 1; i < length; i++) {
      // Don't separate the two halves of a surrogate pair.
      if (Character.isSurrogatePair(sentence.charAt(i - 1), sentence.charAt(i))) {
        continue;
      }
      int score = -this.totalScore;
      if (i - 2 > 0) {
        score += 2 * this.uw1.getOrDefault(sentence.substring(i - 3, i - 2), 0);
      }
      if (i - 1 > 0) {
        score += 2 * this.uw2.getOrDefault(sentence.substring(i - 2, i - 1), 0);
      }
      score += 2 * this.uw3.getOrDefault(sentence.substring(i - 1, i), 0);
      score += 2 * this.uw4.getOrDefault(sentence.substring(i, i + 1), 0);
      if (i + 1 < length) {
        score += 2 * this.uw5.getOrDefault(sentence.substring(i + 1, i + 2), 0);
      }
      if (i + 2 < length) {
        score += 2 * this.uw6.getOrDefault(sentence.substring(i + 2, i + 3), 0);
      }
      if (i > 1) {
        score += 2 * this.bw1.getOrDefault(sentence.substring(i - 2, i), 0);
      }
      score += 2 * this.bw2.getOrDefault(sentence.substring(i - 1, i + 1), 0);
      if (i + 1 < length) {
        score += 2 * this.bw3.getOrDefault(sentence.substring(i, i + 2), 0);
      }
      if (i - 2 > 0) {
        score += 2 * this.tw1.getOrDefault(sentence.substring(i - 3, i), 0);
      }
      if (i - 1 > 0) {
        score += 2 * this.tw2.getOrDefault(sentence.substring(i - 2, i + 1), 0);
      }
      if (i + 1 < length) {
        score += 2 * this.tw3.getOrDefault(sentence.substring(i - 1, i + 2), 0);
      }
      if (i + 2 < length) {
        score += 2 * this.tw4.getOrDefault(sentence.substring(i, i + 3), 0);
      }
      if (score > 0) {
        result.add(sentence.substring(phraseStart, i));
        phraseStart = i;
      }
    }
    result.add(sentence.substring(phraseStart));
    return result;
  }

  /**
   * Translates an HTML string with phrases wrapped in no-breaking markup.
   *
   * @param html an HTML string.
   * @return the translated HTML string with no-breaking markup.
   */
  public String translateHTMLString(String html) {
    String sentence = HTMLProcessor.getText(html);
    List<String> phrases = parse(sentence);
    return HTMLProcessor.resolve(phrases, html, "\u200b");
  }
}
