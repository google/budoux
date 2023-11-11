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
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.NodeVisitor;

/** Processes phrases into an HTML string wrapping them in no-breaking markup. */
final class HTMLProcessor {
  private static final Set<String> skipNodes;
  private static final String STYLE = "word-break: keep-all; overflow-wrap: anywhere;";

  private HTMLProcessor() {}

  static {
    Gson gson = new Gson();
    InputStream inputStream = HTMLProcessor.class.getResourceAsStream("/skip_nodes.json");
    try (Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
      String[] skipNodesStrings = gson.fromJson(reader, String[].class);
      skipNodes = new HashSet<>(Arrays.asList(skipNodesStrings));
    } catch (JsonSyntaxException | JsonIOException | IOException e) {
      throw new AssertionError(e);
    }
  }

  private static class PhraseResolvingNodeVisitor implements NodeVisitor {
    private static final char SEP = '\uFFFF';
    private final String phrasesJoined;
    private final String separator;
    private final StringBuilder output = new StringBuilder();
    private Integer scanIndex = 0;
    private boolean toSkip = false;
    private final ArrayDeque<Boolean> elementStack = new ArrayDeque<>();

    /**
     * Constructs a PhraseResolvingNodeVisitor.
     *
     * @param phrases a list of phrase strings.
     * @param separator the separator string.
     */
    PhraseResolvingNodeVisitor(List<String> phrases, String separator) {
      this.separator = separator;
      this.phrasesJoined = String.join(Character.toString(SEP), phrases);
    }

    /**
     * Returns the resolved output string.
     *
     * @return the output string.
     */
    public StringBuilder getOutput() {
      return output;
    }

    @Override
    public void head(Node node, int depth) {
      if (node.nodeName().equals("body")) {
        return;
      }
      if (node instanceof Element) {
        elementStack.push(toSkip);
        String attributesEncoded =
            node.attributes().asList().stream()
                .map(attribute -> " " + attribute)
                .collect(Collectors.joining(""));
        final String nodeName = node.nodeName();
        if (skipNodes.contains(nodeName.toUpperCase(Locale.ENGLISH))) {
          if (!toSkip && phrasesJoined.charAt(scanIndex) == SEP) {
            output.append(separator);
            scanIndex++;
          }
          toSkip = true;
        }
        output.append(String.format("<%s%s>", nodeName, attributesEncoded));
      } else if (node instanceof TextNode) {
        String data = ((TextNode) node).getWholeText();
        for (int i = 0; i < data.length(); i++) {
          char c = data.charAt(i);
          if (c != phrasesJoined.charAt(scanIndex)) {
            if (!toSkip) {
              output.append(separator);
            }
            scanIndex++;
          }
          scanIndex++;
          output.append(c);
        }
      }
    }

    @Override
    public void tail(Node node, int depth) {
      if (node.nodeName().equals("body") || node instanceof TextNode) {
        return;
      }
      // assume node instanceof Element;
      toSkip = elementStack.pop();
      Element element = (Element) node;
      if (element.tag().isSelfClosing()) {
        return;
      }
      output.append(String.format("</%s>", node.nodeName()));
    }
  }

  /**
   * Wraps phrases in the HTML string with non-breaking markup.
   *
   * @param phrases the phrases included in the HTML string.
   * @param html the HTML string to resolve.
   * @return the HTML string of phrases wrapped in non-breaking markup.
   */
  public static String resolve(List<String> phrases, String html) {
    return resolve(phrases, html, "\u200b");
  }

  /**
   * Wraps phrases in the HTML string with non-breaking markup.
   *
   * @param phrases the phrases included in the HTML string.
   * @param html the HTML string to resolve.
   * @param separator the separator string.
   * @return the HTML string of phrases wrapped in non-breaking markup.
   */
  public static String resolve(List<String> phrases, String html, String separator) {
    Document doc = Jsoup.parseBodyFragment(html);
    PhraseResolvingNodeVisitor nodeVisitor = new PhraseResolvingNodeVisitor(phrases, separator);
    doc.body().traverse(nodeVisitor);
    return String.format("<span style=\"%s\">%s</span>", STYLE, nodeVisitor.getOutput());
  }

  /**
   * Gets the text content from the input HTML string.
   *
   * @param html an HTML string.
   * @return the text content.
   */
  public static String getText(String html) {
    return Jsoup.parseBodyFragment(html).text();
  }
}
