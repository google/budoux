# BudouX for Go

BudouX is a standalone, small, and language-neutral phrase segmenter tool that provides beautiful and legible line breaks.
This is the Go implementation of [BudouX](https://github.com/google/budoux).

## Supported Languages

- Japanese (ja)
- Simplified Chinese (zh-hans)
- Traditional Chinese (zh-hant)
- Thai (th)

## Installation

```bash
go get github.com/google/budoux/golang
```

## Usage

### Simple usage

```go
package main

import (
    "fmt"
    "github.com/google/budoux/golang"
)

func main() {
    parser := budoux.NewDefaultJapaneseParser()
    phrases := parser.Parse("今日は良い天気ですね。")
    fmt.Println(phrases)
    // Output: [今日は 良い 天気ですね。]
}
```

### Supported languages and their default parsers

- Japanese: `budoux.NewDefaultJapaneseParser()`
- Simplified Chinese: `budoux.NewDefaultSimplifiedChineseParser()`
- Traditional Chinese: `budoux.NewDefaultTraditionalChineseParser()`
- Thai: `budoux.NewDefaultThaiParser()`

## API Reference

### Parser

The main interface for text segmentation.

#### Constructor Functions

```go
// Create parsers with default models
func NewDefaultJapaneseParser() *Parser
func NewDefaultSimplifiedChineseParser() *Parser
func NewDefaultTraditionalChineseParser() *Parser
func NewDefaultThaiParser() *Parser

// Create parser with custom model
func New(model models.Model) *Parser
```

#### Methods

```go
// Parse segments input text into phrases
func (p *Parser) Parse(sentence string) []string
```

## Testing

Run the test suite:

```bash
go test ./...
```

Run tests with verbose output:

```bash
go test -v ./...
```

## License

Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
