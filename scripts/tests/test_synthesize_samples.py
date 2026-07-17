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
"""Tests the candidate synthesis script and its granular units."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

# module hack to allow importing scripts and budoux from workspace root
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

# Mock google modules before importing the production script
mock_genai = MagicMock()
sys.modules['google'] = mock_genai
sys.modules['google.genai'] = mock_genai

import budoux  # noqa: E402
from scripts import synthesize_samples  # noqa (module hack)


class TestGetSeparatorIndices(unittest.TestCase):
  """Unit tests for get_separator_indices."""

  def test_standard(self) -> None:
    text_with_seps = f'事態は{budoux.utils.SEP}もはや{budoux.utils.SEP}深刻だ。'
    self.assertEqual(
        synthesize_samples.get_separator_indices(text_with_seps), {3, 6})

  def test_no_separators(self) -> None:
    self.assertEqual(
        synthesize_samples.get_separator_indices('事態はもはや深刻だ。'), set())

  def test_empty(self) -> None:
    self.assertEqual(synthesize_samples.get_separator_indices(''), set())


class TestAlignToBaselineModel(unittest.TestCase):
  """Unit tests for align_to_baseline_model."""

  def test_unsplit_with_buggy_splits(self) -> None:
    # Under buggy model: '事態はもは/や/深刻だ。' (split at 3, 5, 6)
    # Target 'もはや' (starts at 3, ends at 6).
    # LLM broke at 3 and 6: '事態は▁もはや▁深刻だ。'
    clean_text = '事態はもはや深刻だ。'
    baseline_breaks = {0, 3, 5, 6, 10}
    llm_breaks = {3, 6}
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='もはや',
        split_offset=None)
    # The buggy internal split at index 5 must be discarded,
    # and outer boundaries 3 and 6 should be forced because they are in llm_breaks.
    self.assertEqual(aligned, f'事態は{budoux.utils.SEP}もはや{budoux.utils.SEP}深刻だ。')

  def test_unsplit_already_unsplit(self) -> None:
    # Baseline has correct outer splits and no internal splits.
    clean_text = '事態はもはや深刻だ。'
    baseline_breaks = {0, 3, 6, 10}
    llm_breaks = {3, 6}
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='もはや',
        split_offset=None)
    self.assertEqual(aligned, f'事態は{budoux.utils.SEP}もはや{budoux.utils.SEP}深刻だ。')

  def test_unsplit_at_sentence_start(self) -> None:
    # Target phrase is at the start of sentence (idx = 0).
    clean_text = 'もはや深刻だ。'
    baseline_breaks = {0, 7}
    llm_breaks = {3}  # Only split after 'もはや'
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='もはや',
        split_offset=None)
    # Outer split before 'もはや' is skipped because it's at index 0
    self.assertEqual(aligned, f'もはや{budoux.utils.SEP}深刻だ。')

  def test_unsplit_with_particle(self) -> None:
    # Target 'ありがとう' is followed by particle 'と': 'ありがとうと言った。' (idx 0-5)
    # LLM output: 'ありがとうと▁言った。' -> no split at index 5, split at index 6.
    # Baseline parser predicted: 'ありが/とう/と/言った。' -> splits at {3, 5, 6, 10}
    clean_text = 'ありがとうと言った。'
    baseline_breaks = {0, 3, 5, 6, 10}
    llm_breaks = {6}
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='ありがとう',
        split_offset=None)
    # Buggy internal splits (3) are discarded.
    # Split at 5 (between 'ありがとう' and 'と') is discarded because it is not in llm_breaks.
    # Background split at 6 (between 'と' and '言った') is preserved.
    self.assertEqual(aligned, f'ありがとうと{budoux.utils.SEP}言った。')

  def test_unsplit_force_boundary_split(self) -> None:
    # Target 'ありがとう' followed by word '感謝': 'ありがとう感謝。' (idx 0-5)
    # LLM output: 'ありがとう▁感謝。' -> split at 5.
    clean_text = 'ありがとう感謝。'
    baseline_breaks = {0}
    llm_breaks = {5}
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='ありがとう',
        split_offset=None)
    # Split at index 5 is forced because LLM chose to split there, even if baseline had no split.
    self.assertEqual(aligned, f'ありがとう{budoux.utils.SEP}感謝。')

  def test_split_standard(self) -> None:
    # Positive sample for transition 'もは' in '誰もはっきりと答えない。'
    clean_text = '誰もはっきりと答えない。'
    baseline_breaks = {0, 7, 12}
    llm_breaks = {2}  # LLM split after 'も'
    aligned = synthesize_samples.align_to_baseline_model(
        clean_text=clean_text,
        baseline_breaks=baseline_breaks,
        llm_breaks=llm_breaks,
        target_sequence='もは',
        split_offset=1)
    # Should force break between 'も' and 'は' (index 2) and preserve baseline break at index 7.
    self.assertEqual(aligned,
                     f'誰も{budoux.utils.SEP}はっきりと{budoux.utils.SEP}答えない。')


class TestValidateNegativeCandidate(unittest.TestCase):
  """Unit tests for validate_negative_candidate."""

  def test_valid_unsplit(self) -> None:
    candidate = f'事態は{budoux.utils.SEP}もはや{budoux.utils.SEP}深刻だ。'
    self.assertTrue(
        synthesize_samples.validate_negative_candidate(candidate, 'もはや'))

  def test_invalid_split_inside(self) -> None:
    candidate = f'事態は{budoux.utils.SEP}もは{budoux.utils.SEP}や{budoux.utils.SEP}深刻だ。'
    self.assertFalse(
        synthesize_samples.validate_negative_candidate(candidate, 'もはや'))


class TestValidatePositiveCandidate(unittest.TestCase):
  """Unit tests for validate_positive_candidate."""

  def test_valid_split(self) -> None:
    candidate = f'誰も{budoux.utils.SEP}はっきりと答えない。'
    self.assertTrue(
        synthesize_samples.validate_positive_candidate(candidate, 'も', 'は',
                                                       ['もはや']))

  def test_invalid_no_split(self) -> None:
    candidate = '誰もはっきりと答えない。'
    self.assertFalse(
        synthesize_samples.validate_positive_candidate(candidate, 'も', 'は',
                                                       ['もはや']))

  def test_invalid_contaminated(self) -> None:
    candidate = f'私は{budoux.utils.SEP}もはや{budoux.utils.SEP}限界だ。'
    self.assertFalse(
        synthesize_samples.validate_positive_candidate(candidate, 'も', 'は',
                                                       ['もはや']))


class TestParseIssue(unittest.TestCase):
  """Unit tests for parse_issue."""

  @patch('requests.get')
  def test_parse_issue_happy_path(self, mock_get: MagicMock) -> None:
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {'body': '   もはや / これまでべきが細切れに   '}
    mock_get.return_value = mock_res

    sanitized = synthesize_samples.parse_issue('841')
    self.assertEqual(sanitized, 'もはや / これまでべきが細切れに')

  @patch('requests.get')
  def test_parse_issue_connection_error(self, mock_get: MagicMock) -> None:
    mock_get.side_effect = requests.exceptions.ConnectionError('Network down')
    with self.assertRaises(SystemExit):
      synthesize_samples.parse_issue('841')


class TestSynthesize(unittest.TestCase):
  """Integration verification for LLM response integration and filtering orchestration."""

  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.api_key = 'fake_key_123'

    # A tiny baseline model structured specifically to verify background chunk alignment and bug recovery:
    # - TW1: splits after '事態は' and '今日は'
    # - UW4: splits before '深', '答', 'あ'
    # - BW2: buggy feature causing over-segmentation inside 'もはや' ('もは/や')
    # - UW1: balances base_score to -1500
    self.tiny_model = {
        'TW1': {
            '事態は': 3000,
            '今日は': 3000
        },
        'UW4': {
            '深': 3000,
            '答': 3000,
            'あ': 3000
        },
        'BW2': {
            'はや': 3000
        },
        'UW1': {
            'dummy': -15000
        }
    }
    self.parser = budoux.Parser(self.tiny_model)

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  @patch('scripts.synthesize_samples.genai')
  def test_synthesize_happy_path(self, mock_genai_module: MagicMock) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    parsed_response = synthesize_samples.SynthesisResponse(
        analysis=synthesize_samples.Analysis(
            negative_phrases=['もはや'],
            positive_phrases=['いよいよ/はじまる'],
        ),
        negative_sentences=[
            synthesize_samples.NegativeSentenceGroup(
                phrase='もはや',
                sentences=[
                    '事態は▁もはや▁深刻だ。',
                    '彼の▁実力は▁もは▁や▁止められない。',  # Invalid: contains split inside "もはや"
                    '本日は▁とても▁良い▁天気だ。'  # Invalid: missing target phrase "もはや"
                ])
        ],
        positive_sentences=[
            synthesize_samples.PositiveSentenceGroup(
                char1='も',
                char2='は',
                sentences=[
                    '誰も▁はっきりと答えない。',
                    '私は▁もはや▁限界だ。'  # Invalid: contaminated with unsplit phrase "もはや"
                ]),
            synthesize_samples.PositiveSentenceGroup(
                char1='は',
                char2='や',
                sentences=[
                    '今日は▁やるべきことがある。',
                    '部屋に▁入る。'  # Invalid: does not contain target transition "は" + "や"
                ]),
            synthesize_samples.PositiveSentenceGroup(
                char1='よ', char2='は', sentences=[
                    'いよいよ▁はじまる。',
                ])
        ])
    mock_response.parsed = parsed_response
    mock_client.models.generate_content.return_value = mock_response
    mock_genai_module.Client.return_value = mock_client

    pos, neg, neg_phrases, pos_phrases = synthesize_samples.synthesize(
        '「もはや」が細切れになってしまう。あと、「いよいよはじまる」も「いよいよ」の後で切れるべき。',
        self.api_key,
        self.parser,
        num_samples=5,
    )

    self.assertEqual(neg_phrases, ['もはや'])
    self.assertEqual(pos_phrases, ['いよいよ/はじまる'])

    # Verify positive samples preserve realistic background breaks while strictly splitting targeted transitions
    self.assertEqual(pos, [
        f'誰も{budoux.utils.SEP}はっきりと{budoux.utils.SEP}答えない。',
        f'今日は{budoux.utils.SEP}やるべきことが{budoux.utils.SEP}ある。',
        f'いよいよ{budoux.utils.SEP}はじまる。'
    ])

    # Verify negative sample eliminates internal over-segmentation defect
    self.assertEqual(neg, [f'事態は{budoux.utils.SEP}もはや{budoux.utils.SEP}深刻だ。'])

    # Verify default model parameter passed to API client
    mock_client.models.generate_content.assert_called_once_with(
        model='gemini-3.1-flash-lite',
        contents=unittest.mock.ANY,
        config=unittest.mock.ANY,
    )

  @patch('scripts.synthesize_samples.genai')
  def test_synthesize_custom_model(self, mock_genai_module: MagicMock) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = synthesize_samples.SynthesisResponse(
        analysis=synthesize_samples.Analysis(
            negative_phrases=[], positive_phrases=[]),
        negative_sentences=[],
        positive_sentences=[],
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_genai_module.Client.return_value = mock_client

    synthesize_samples.synthesize(
        'Fake issue context',
        self.api_key,
        self.parser,
        model='gemini-2.0-pro-experimental',
    )
    mock_client.models.generate_content.assert_called_once_with(
        model='gemini-2.0-pro-experimental',
        contents=unittest.mock.ANY,
        config=unittest.mock.ANY,
    )

  @patch('scripts.synthesize_samples.genai')
  def test_synthesize_api_error(self, mock_genai_module: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        'Quota exceeded')
    mock_genai_module.Client.return_value = mock_client
    with self.assertRaises(SystemExit):
      synthesize_samples.synthesize('Fake Issue Body', self.api_key,
                                    self.parser)


if __name__ == '__main__':
  unittest.main()
