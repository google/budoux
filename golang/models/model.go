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

package models

type Model map[string]map[string]float64

// TotalScore calculates the total score of the model.
func (m Model) TotalScore() float64 {
	score := 0.0
	for _, g := range m {
		for _, s := range g {
			score += s
		}
	}
	return score
}

func GetDefaultJapaneseModel() Model {
	return ja
}

func GetDefaultThaiModel() Model {
	return th
}

func GetDefaultSimplifiedChineseModel() Model {
	return zhhans
}

func GetDefaultTraditionalChineseModel() Model {
	return zhhant
}
