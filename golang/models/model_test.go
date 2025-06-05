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

import (
	"fmt"
	"testing"
)

func TestModelTotalScore(t *testing.T) {
	cases := []struct {
		Expected float64
		Model    Model
	}{
		{
			Expected: 10,
			Model: Model{
				"good": {
					"morning":   1,
					"afternoon": 2,
				},
				"bad": {
					"morning":   3,
					"afternoon": 4,
				},
			},
		},
	}
	for i, c := range cases {
		t.Run(fmt.Sprintf("case %d", i), func(t *testing.T) {
			score := c.Model.TotalScore()
			if score != c.Expected {
				t.Errorf("Expected %f, got %f", c.Expected, score)
			}
		})
	}
}
