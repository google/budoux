/*
 * Copyright 2026 Google LLC
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
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Shared declarative HTML and character specification loaded from {@code /html_spec.json}. */
final class HtmlSpec {
  static final HtmlSpec INSTANCE = load();

  static final class ElementActions {
    List<String> block;
    List<String> skip;
    List<String> nobreak;

    @SerializedName("break")
    List<String> breakElements;

    @SerializedName("break_opportunity")
    List<String> breakOpportunity;
  }

  @SerializedName("element_actions")
  ElementActions elementActions;

  @SerializedName("void_elements")
  List<String> voidElements;

  @SerializedName("invisible_format_chars")
  List<String> invisibleFormatChars;

  @SerializedName("space_chars")
  List<String> spaceChars;

  transient Set<String> blockNodes;
  transient Set<String> skipNodes;
  transient Set<String> noBreakNodes;
  transient Set<String> breakNodes;
  transient Set<String> breakOpportunityNodes;
  transient Set<Integer> invisibleFormatCodePoints;
  transient Set<Integer> spaceCodePoints;

  private static HtmlSpec load() {
    Gson gson = new Gson();
    InputStream inputStream = HtmlSpec.class.getResourceAsStream("/html_spec.json");
    if (inputStream == null) {
      throw new AssertionError("Resource not found: /html_spec.json");
    }
    try (Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
      HtmlSpec spec = gson.fromJson(reader, HtmlSpec.class);
      spec.blockNodes = Collections.unmodifiableSet(new HashSet<>(spec.elementActions.block));
      spec.skipNodes = Collections.unmodifiableSet(new HashSet<>(spec.elementActions.skip));
      spec.noBreakNodes = Collections.unmodifiableSet(new HashSet<>(spec.elementActions.nobreak));
      spec.breakNodes =
          Collections.unmodifiableSet(new HashSet<>(spec.elementActions.breakElements));
      spec.breakOpportunityNodes =
          Collections.unmodifiableSet(new HashSet<>(spec.elementActions.breakOpportunity));

      Set<Integer> invisibleCps = new HashSet<>();
      for (String s : spec.invisibleFormatChars) {
        invisibleCps.add(s.codePointAt(0));
      }
      spec.invisibleFormatCodePoints = Collections.unmodifiableSet(invisibleCps);

      Set<Integer> spaceCps = new HashSet<>();
      for (String s : spec.spaceChars) {
        spaceCps.add(s.codePointAt(0));
      }
      spec.spaceCodePoints = Collections.unmodifiableSet(spaceCps);
      return spec;
    } catch (JsonSyntaxException | JsonIOException | IOException e) {
      throw new AssertionError(e);
    }
  }
}
