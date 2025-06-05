// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package budoux

import "github.com/google/budoux/golang/models"

type Parser struct {
	model models.Model
}

// Create new parser.
func New(model models.Model) *Parser {
	return &Parser{
		model: model,
	}
}

// NewDefaultJapaneseParser returns new Parser with default japanese model.
func NewDefaultJapaneseParser() *Parser {
	return New(models.GetDefaultJapaneseModel())
}

// NewDefaultThaiParser returns new Parser with default thai model.
func NewDefaultThaiParser() *Parser {
	return New(models.GetDefaultThaiModel())
}

// NewDefaultSimplifiedChineseParser returns new Parser with default simplified chinese model.
func NewDefaultSimplifiedChineseParser() *Parser {
	return New(models.GetDefaultSimplifiedChineseModel())
}

// NewDefaultTraditionalChineseParser returns new Parser with default traditional chinese model.
func NewDefaultTraditionalChineseParser() *Parser {
	return New(models.GetDefaultTraditionalChineseModel())
}

// getScore returns the score of a word in a group if exists.
// otherwise returns defaultScore.
func (p *Parser) getScore(group string, word string, defaultScore float64) float64 {
	if g, ok := p.model[group]; ok {
		if score, ok := g[word]; ok {
			//return float64(score)
			return score
		}
	}
	return defaultScore
}

// Parses a sentence into phrases.
func (p *Parser) Parse(sentence string) []string {
	if sentence == "" {
		return []string{}
	}
	runes := []rune(sentence)
	//baseScore := -float64(p.model.TotalScore()) * 0.5
	baseScore := -p.model.TotalScore() * 0.5
	chunks := []string{string(runes[0])}
	for i := 1; i < len(runes); i++ {
		score := baseScore
		if i > 2 {
			score += p.getScore("UW1", string(runes[i-3]), 0)
		}
		if i > 1 {
			score += p.getScore("UW2", string(runes[i-2]), 0)
		}
		score += p.getScore("UW3", string(runes[i-1]), 0)
		score += p.getScore("UW4", string(runes[i]), 0)
		if i+1 < len(runes) {
			score += p.getScore("UW5", string(runes[i+1]), 0)
		}
		if i+2 < len(runes) {
			score += p.getScore("UW6", string(runes[i+2]), 0)
		}

		if i > 1 {
			score += p.getScore("BW1", string(runes[i-2:i]), 0)
		}
		score += p.getScore("BW2", string(runes[i-1:i+1]), 0)
		if i+1 < len(runes) {
			score += p.getScore("BW3", string(runes[i:i+2]), 0)
		}

		if i > 2 {
			score += p.getScore("TW1", string(runes[i-3:i]), 0)
		}
		if i > 1 {
			score += p.getScore("TW2", string(runes[i-2:i+1]), 0)
		}
		if i+1 < len(runes) {
			score += p.getScore("TW3", string(runes[i-1:i+2]), 0)
		}
		if i+2 < len(runes) {
			score += p.getScore("TW4", string(runes[i:i+3]), 0)
		}

		if score > 0 {
			chunks = append(chunks, string(runes[i]))
		} else {
			chunks[len(chunks)-1] += string(runes[i])
		}
	}
	return chunks
}
