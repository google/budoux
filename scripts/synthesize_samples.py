# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Candidate sample synthesis utility using Gemini from GitHub issues."""

import argparse
import os
import sys
import typing

import pydantic
import requests
from google import genai

# module hack to allow importing budoux from parent directory
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import budoux  # noqa: E402


class Analysis(pydantic.BaseModel):
  negative_phrases: typing.List[str]
  positive_phrases: typing.List[str]


class NegativeSentenceGroup(pydantic.BaseModel):
  phrase: str
  sentences: typing.List[str]


class PositiveSentenceGroup(pydantic.BaseModel):
  char1: str
  char2: str
  sentences: typing.List[str]


class SynthesisResponse(pydantic.BaseModel):
  analysis: Analysis
  negative_sentences: typing.List[NegativeSentenceGroup]
  positive_sentences: typing.List[PositiveSentenceGroup]


def parse_issue(issue_id: str) -> str:
  """Fetches and sanitizes a GitHub issue body.

  Args:
    issue_id: The GitHub issue ID to fetch.

  Returns:
    The sanitized issue body text.
  """
  url = f'https://api.github.com/repos/google/budoux/issues/{issue_id}'

  try:
    res = requests.get(url, timeout=10)
  except requests.exceptions.RequestException as e:
    sys.exit(f'Error: Failed to connect to GitHub API: {e}')

  if res.status_code != 200:
    sys.exit(
        f'Error: Failed to fetch GitHub issue #{issue_id} (status: {res.status_code})'
    )
  body = res.json().get('body', '')
  return typing.cast(str, body).strip()


def get_separator_indices(text_with_seps: str) -> typing.Set[int]:
  """Computes the character boundary indices where separators are inserted.

  Args:
    text_with_seps: String containing separator characters.

  Returns:
    A set of character indices representing splits in the plain text.
  """
  indices = set()
  curr_idx = 0
  for char in text_with_seps:
    if char == budoux.utils.SEP:
      indices.add(curr_idx)
    else:
      curr_idx += 1
  return indices


def align_to_baseline_model(
    clean_text: str,
    baseline_breaks: typing.Set[int],
    llm_breaks: typing.Set[int],
    target_sequence: str,
    split_offset: typing.Optional[int] = None,
) -> str:
  """Aligns background phrase boundaries to baseline model predictions with target overrides.

  This function overrides baseline boundary splits within and around a target sequence:
  1. Forces target_sequence to be unbroken internally, except at split_offset if provided.
  2. Syncs the splits immediately before and after target_sequence with the LLM's choices
     (llm_breaks), allowing context-dependent boundary splits (e.g. keeping particles attached).
  3. Preserves baseline breaks for all other background positions.

  Args:
    clean_text: The sentence plain text without any separators.
    baseline_breaks: Set of boundary indices predicted by the baseline model.
    llm_breaks: Set of boundary indices chosen by the LLM in its raw output.
    target_sequence: The exact substring in clean_text to apply overrides to.
    split_offset: Offset within target_sequence where a split must be forced.
      If None, the entire target_sequence is kept completely unsplit.

  Returns:
    The reconstructed sentence containing aligned separator characters.
  """
  predicted_breaks = set(baseline_breaks)

  # Apply overrides on the target sequence local neighborhood
  idx = clean_text.find(target_sequence)
  if idx != -1:
    target_len = len(target_sequence)
    # Clear any baseline splits inside the target sequence
    for i in range(idx + 1, idx + target_len):
      predicted_breaks.discard(i)

    # Force the split at the target offset if requested
    if split_offset is not None:
      predicted_breaks.add(idx + split_offset)

    # Sync splits immediately before and after the target sequence with the LLM's choices
    if idx > 0:
      if idx in llm_breaks:
        predicted_breaks.add(idx)
      else:
        predicted_breaks.discard(idx)

    if idx + target_len < len(clean_text):
      outer_end_idx = idx + target_len
      if outer_end_idx in llm_breaks:
        predicted_breaks.add(outer_end_idx)
      else:
        predicted_breaks.discard(outer_end_idx)

  # Reconstruct the sentence with separators inserted at the break indices
  result = []
  for i, c in enumerate(clean_text):
    if i in predicted_breaks and i > 0:
      result.append(budoux.utils.SEP)
    result.append(c)
  return ''.join(result)


def generate_synthesis_prompt(issue_context: str, num_samples: int) -> str:
  """Generates the LLM synthesis prompt based on the issue context and sample count.

  Args:
    issue_context: Body of the GitHub issue.
    num_samples: Number of candidates to generate.

  Returns:
    The prompt string.
  """
  return f"""You are an expert Japanese linguist training a BudouX segmenter model.
Analyze the following GitHub issue bug report:
{issue_context}

Your task is to analyze the issue and identify:
1. "negative_phrases": List of phrases/compounds that were incorrectly split internally and should remain as a single unsplit unit (e.g. adverbs like "もはや" or compounds like "こんにちは").
2. "positive_phrases": List of phrases containing a split separator "/" representing boundaries that were incorrectly kept unsplit and MUST be split (e.g. "いよいよ/はじまる" meaning a split is forced between "いよいよ" and "はじまる").

Then, generate natural Japanese training sentences for each:
- For each negative phrase (e.g. "もはや"): Generate {num_samples} natural Japanese sentences where the phrase is completely unsplit internally, but has splits before and after it. Mark the boundaries before and after with the separator "▁" (e.g. "もはや▁これまで" or "技術は▁もはや▁時代遅れ").
- For each adjacent character transition within the negative phrases (for example, if the phrase is "もはや", the transitions are "も" and "は", and "は" and "や"): Generate {num_samples} natural Japanese sentences where the two characters appear adjacent but split across a semantic boundary (e.g. "私も▁はやく走る" or "いつも▁はれる"). Do NOT include the negative phrases themselves in these sentences.
- For each positive phrase containing a "/" boundary (e.g. "いよいよ/はじまる"): Generate {num_samples} natural Japanese sentences where the boundary is split. Mark the split with the separator "▁" (e.g. "いよいよ▁はじまる。").
CRITICAL RULES FOR TRANSITION SENTENCES:
1. The two characters of the transition ('char1' and 'char2') MUST appear immediately adjacent as literal characters in the written text of the sentence. Do NOT confuse pronunciation or hiragana readings with written characters. For example, if the transition is 'も' and 'は', the character 'は' must literally appear in the Japanese text (e.g. 'もはっきり'); a kanji like '晴' pronounced 'は' is INVALID.
2. The split separator '▁' must be placed EXACTLY between 'char1' and 'char2' (e.g. '私も▁はやく').
3. The split separator '▁' MUST represent a valid word or phrase boundary. You MUST NOT split a single word internally (e.g. do NOT split "ありえない" as "あ▁りえない", or "ありがたい" as "あ▁りがたい", or "ございます" as "ご▁ざいます"). The split must only occur between two separate words, particles, or clauses.
Format your output strictly as a JSON object matching this schema:
{{
  "analysis": {{
    "negative_phrases": ["もはや"],
    "positive_phrases": ["いよいよ/はじまる"]
  }},
  "negative_sentences": [
    {{
      "phrase": "もはや",
      "sentences": [
        "もはや▁これまでだ。",
        "技術は▁もはや▁時代遅れだ。"
      ]
    }}
  ],
  "positive_sentences": [
    {{
      "char1": "も",
      "char2": "は",
      "sentences": [
        "私も▁はやく走りたい。",
        "いつも▁はれる。"
      ]
    }},
    {{
      "char1": "は",
      "char2": "や",
      "sentences": [
        "これは▁やはり面白い。",
        "うちわは▁やはり便利だ。"
      ]
    }},
    {{
      "char1": "よ",
      "char2": "は",
      "sentences": [
        "試合がいよいよ▁はじまる。"
      ]
    }}
  ]
}}

Ensure all sentences are grammatically correct and natural Japanese.
Do not include any other markdown formatting outside the raw JSON object.
"""


def get_synthesis_config() -> genai.types.GenerateContentConfig:
  """Configures the structured output using Pydantic model.

  Returns:
    The GenerateContentConfig object.
  """
  return genai.types.GenerateContentConfig(
      response_mime_type='application/json',
      response_schema=SynthesisResponse,
  )


def validate_negative_candidate(candidate_with_breaks: str,
                                target_phrase: str) -> bool:
  """Validates that a negative candidate doesn't have internal splits within target_phrase.

  Args:
    candidate_with_breaks: Sentence generated by LLM, possibly containing breaks.
    target_phrase: The phrase that must remain unsplit.

  Returns:
    True if the candidate contains no internal breaks inside the phrase.
  """
  internal_char_pairs = [
      target_phrase[i:i + 2] for i in range(len(target_phrase) - 1)
  ]
  for pair in internal_char_pairs:
    if f'{pair[0]}{budoux.utils.SEP}{pair[1]}' in candidate_with_breaks:
      return False
  return True


def validate_positive_candidate(
    candidate_with_breaks: str,
    left_char: str,
    right_char: str,
    negative_phrases: typing.List[str],
) -> bool:
  """Validates that a positive candidate is split at the transition and has no parent contamination.

  Args:
    candidate_with_breaks: Sentence generated by LLM, possibly containing breaks.
    left_char: Character before the transition.
    right_char: Character after the transition.
    negative_phrases: List of negative target phrases to prevent contamination.

  Returns:
    True if the candidate is clean and correctly split.
  """
  for phrase in negative_phrases:
    if phrase in candidate_with_breaks:
      return False
  return f'{left_char}{budoux.utils.SEP}{right_char}' in candidate_with_breaks


def filter_and_align_negatives(
    negative_sentences_data: typing.List[NegativeSentenceGroup],
    negative_phrases: typing.List[str],
    parser: budoux.Parser,
) -> typing.List[str]:
  """Filters and aligns raw LLM negative candidate sentences.

  Args:
    negative_sentences_data: List of negative candidate groups.
    negative_phrases: List of target negative phrases to align.
    parser: Baseline parser for predicting background breaks.

  Returns:
    List of aligned negative training sentences.
  """
  results = []
  for item in negative_sentences_data:
    phrase = item.phrase
    if phrase not in negative_phrases:
      continue
    for llm_candidate in item.sentences:
      clean_text = llm_candidate.replace(budoux.utils.SEP, '')
      if phrase not in clean_text:
        continue

      if not validate_negative_candidate(llm_candidate, phrase):
        continue

      baseline_breaks = get_separator_indices(
          budoux.utils.SEP.join(parser.parse(clean_text)))
      llm_breaks = get_separator_indices(llm_candidate)
      aligned = align_to_baseline_model(
          clean_text=clean_text,
          baseline_breaks=baseline_breaks,
          llm_breaks=llm_breaks,
          target_sequence=phrase,
          split_offset=None,
      )
      results.append(aligned)
  return results


def filter_and_align_positives(
    positive_sentences_data: typing.List[PositiveSentenceGroup],
    negative_phrases: typing.List[str],
    parser: budoux.Parser,
) -> typing.List[str]:
  """Filters and aligns raw LLM positive candidate sentences.

  Args:
    positive_sentences_data: List of split transition candidate objects.
    negative_phrases: List of negative target phrases to prevent contamination.
    parser: Baseline parser for predicting background breaks.

  Returns:
    List of aligned positive training sentences.
  """
  results = []
  for item in positive_sentences_data:
    left_char = item.char1
    right_char = item.char2
    target_transition = left_char + right_char
    for llm_candidate in item.sentences:
      clean_text = llm_candidate.replace(budoux.utils.SEP, '')
      if target_transition not in clean_text:
        continue

      if not validate_positive_candidate(llm_candidate, left_char, right_char,
                                         negative_phrases):
        continue

      baseline_breaks = get_separator_indices(
          budoux.utils.SEP.join(parser.parse(clean_text)))
      llm_breaks = get_separator_indices(llm_candidate)
      aligned = align_to_baseline_model(
          clean_text=clean_text,
          baseline_breaks=baseline_breaks,
          llm_breaks=llm_breaks,
          target_sequence=target_transition,
          split_offset=len(left_char),
      )
      results.append(aligned)
  return results


def synthesize(
    issue_context: str,
    api_key: str,
    parser: budoux.Parser,
    num_samples: int = 15,
    model: str = 'gemini-3.1-flash-lite',
) -> typing.Tuple[
    typing.List[str], typing.List[str], typing.List[str], typing.List[str]
]:
  """Generates positive and negative Japanese training sentences using Gemini based on a GitHub issue.

  Args:
    issue_context: The raw text/body of the GitHub issue context.
    api_key: Gemini API developer credentials key.
    parser: The baseline BudouX Parser to align background separators against.
    num_samples: The number of candidate sentences to generate per phrase or transition.
    model: The Gemini model ID to use for content generation.

  Returns:
    A tuple containing four lists:
      (positive_sentences, negative_sentences, negative_phrases, positive_phrases).
  """
  client = genai.Client(api_key=api_key)
  prompt = generate_synthesis_prompt(issue_context, num_samples)
  config = get_synthesis_config()

  try:
    response = client.models.generate_content(
        model=model, contents=prompt, config=config)
  except Exception as e:
    sys.exit(f'Error: Gemini API call failed: {e}')

  result_data = response.parsed
  if not isinstance(result_data, SynthesisResponse):
    sys.exit('Error: Gemini API returned empty or unparseable response.')

  negative_phrases = result_data.analysis.negative_phrases
  positive_phrases = result_data.analysis.positive_phrases

  pos_aligned = []
  neg_aligned = []

  # 1. Filter and align negative sentences for target phrases
  neg_aligned.extend(
      filter_and_align_negatives(
          negative_sentences_data=result_data.negative_sentences,
          negative_phrases=negative_phrases,
          parser=parser,
      ))

  # 2. Filter and align positive sentences for forced splits
  pos_aligned.extend(
      filter_and_align_positives(
          positive_sentences_data=result_data.positive_sentences,
          negative_phrases=negative_phrases,
          parser=parser,
      ))

  return (
      list(dict.fromkeys(pos_aligned)),
      list(dict.fromkeys(neg_aligned)),
      negative_phrases,
      positive_phrases,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '-i',
      '--issue',
      required=True,
      help='GitHub issue ID to fetch context and target definitions from.')
  parser.add_argument(
      '-n',
      '--num-samples',
      type=int,
      default=15,
      help='Number of training samples to synthesize (default: 15).')
  parser.add_argument(
      '-o',
      '--output',
      default='staging_raw.txt',
      help='Output staged file path (default: staging_raw.txt).')
  parser.add_argument(
      '-m',
      '--model',
      default='gemini-3.1-flash-lite',
      help='Gemini model to use (default: gemini-3.1-flash-lite).')
  args = parser.parse_args()

  api_key = os.environ.get('GEMINI_API_KEY')
  if not api_key:
    sys.exit('Error: GEMINI_API_KEY environment variable is not set.')

  # Load the current default parser for Japanese to align background separators
  parser_instance = budoux.load_default_japanese_parser()

  issue_context = parse_issue(args.issue)

  pos, neg, _, _ = synthesize(
      issue_context,
      api_key,
      parser_instance,
      args.num_samples,
      args.model,
  )

  # Write output candidates to staging file
  with open(args.output, 'w', encoding='utf-8') as f:
    for sent in pos + neg:
      f.write(sent + '\n')

  print(
      f'Candidate synthesis complete. Staged {len(pos)} positive and {len(neg)} negative rows in {args.output}.'
  )


if __name__ == '__main__':
  main()
