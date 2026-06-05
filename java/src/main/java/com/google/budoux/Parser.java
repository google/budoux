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
  private final Map<String, Map<String, Integer>> model;

  /**
   * Constructs a BudouX parser.
   *
   * @param model the model data.
   */
  public Parser(Map<String, Map<String, Integer>> model) {
    this.model = model;
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
   * Gets the score for the specified feature of the given sequence.
   *
   * @param featureKey the feature key to examine.
   * @param sequence the sequence to look up the score.
   * @return the contribution score to support a phrase break.
   */
  private int getScore(String featureKey, String sequence) {
    Map<String, Integer> group = this.model.get(featureKey);
    return group != null ? group.getOrDefault(sequence, 0) : 0;
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
    result.add(String.valueOf(sentence.charAt(0)));
    int totalScore =
        this.model.values().stream()
            .mapToInt(group -> group.values().stream().mapToInt(Integer::intValue).sum())
            .sum();
    Map<String, Integer> uw1 = this.model.get("UW1");
    Map<String, Integer> uw2 = this.model.get("UW2");
    Map<String, Integer> uw3 = this.model.get("UW3");
    Map<String, Integer> uw4 = this.model.get("UW4");
    Map<String, Integer> uw5 = this.model.get("UW5");
    Map<String, Integer> uw6 = this.model.get("UW6");
    Map<String, Integer> bw1 = this.model.get("BW1");
    Map<String, Integer> bw2 = this.model.get("BW2");
    Map<String, Integer> bw3 = this.model.get("BW3");
    Map<String, Integer> tw1 = this.model.get("TW1");
    Map<String, Integer> tw2 = this.model.get("TW2");
    Map<String, Integer> tw3 = this.model.get("TW3");
    Map<String, Integer> tw4 = this.model.get("TW4");
    for (int i = 1; i < sentence.length(); i++) {
      int score = -totalScore;
      if (i - 2 > 0 && uw1 != null) {
        score += 2 * uw1.getOrDefault(sentence.substring(i - 3, i - 2), 0);
      }
      if (i - 1 > 0 && uw2 != null) {
        score += 2 * uw2.getOrDefault(sentence.substring(i - 2, i - 1), 0);
      }
      if (uw3 != null) {
        score += 2 * uw3.getOrDefault(sentence.substring(i - 1, i), 0);
      }
      if (uw4 != null) {
        score += 2 * uw4.getOrDefault(sentence.substring(i, i + 1), 0);
      }
      if (i + 1 < sentence.length() && uw5 != null) {
        score += 2 * uw5.getOrDefault(sentence.substring(i + 1, i + 2), 0);
      }
      if (i + 2 < sentence.length() && uw6 != null) {
        score += 2 * uw6.getOrDefault(sentence.substring(i + 2, i + 3), 0);
      }
      if (i > 1 && bw1 != null) {
        score += 2 * bw1.getOrDefault(sentence.substring(i - 2, i), 0);
      }
      if (bw2 != null) {
        score += 2 * bw2.getOrDefault(sentence.substring(i - 1, i + 1), 0);
      }
      if (i + 1 < sentence.length() && bw3 != null) {
        score += 2 * bw3.getOrDefault(sentence.substring(i, i + 2), 0);
      }
      if (i - 2 > 0 && tw1 != null) {
        score += 2 * tw1.getOrDefault(sentence.substring(i - 3, i), 0);
      }
      if (i - 1 > 0 && tw2 != null) {
        score += 2 * tw2.getOrDefault(sentence.substring(i - 2, i + 1), 0);
      }
      if (i + 1 < sentence.length() && tw3 != null) {
        score += 2 * tw3.getOrDefault(sentence.substring(i - 1, i + 2), 0);
      }
      if (i + 2 < sentence.length() && tw4 != null) {
        score += 2 * tw4.getOrDefault(sentence.substring(i, i + 3), 0);
      }
      if (score > 0) {
        result.add("");
      }
      result.set(result.size() - 1, result.get(result.size() - 1) + sentence.charAt(i));
    }
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
