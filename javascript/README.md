# BudouX JavaScript module

BudouX is a standalone, small, and language-neutral phrase segmenter tool that provides beautiful and legible line breaks.

For more details about the project, please refer to the [project README](https://github.com/google/budoux/README.md).

## Install
```
npm install ./javascript
```

## Usage

### Simple usage
You can get a list of phrases by feeding a script to the parser.

```javascript
import { loadDefaultJapaneseParser } from 'budoux';
const parser = loadDefaultJapaneseParser();
console.log(parser.parse('今日は天気です。'));
// ['今日は', '天気です。']
```

### Translating an HTML string
You can also translate an HTML string to wrap phrases with non-breaking markup.
```javascript
console.log(parser.translateHTMLString('今日は<b>とても天気</b>です。'));
// <span style="word-break: keep-all; overflow-wrap: break-word;">今日は<b><wbr>とても<wbr>天気</b>です。</span>
```

### Applying to an HTML element
You can also feed an HTML element to the parser to apply the process.
```javascript
const ele = document.querySelector('p.budou-this');
console.log(ele.outerHTML);
// <p class="budou-this">今日は<b>とても天気</b>です。</p>
parser.applyElement(ele);
console.log(ele.outerHTML);
// <p class="budou-this" style="word-break: keep-all; overflow-wrap: break-word;">今日は<b><wbr>とても<wbr>天気</b>です。</p>
```

### Loading a custom model
You can load your own custom model as follows.
```javascript
import { Parser } from 'budoux';
const model = JSON.parse('{"BB2:108120": 1817}');  // Content of the custom model JSON file.
const parser = new Parser(model);
```

## Caveat
BudouX supports HTML inputs and outputs HTML strings with markup applied to wrap phrases, but it's not meant to be used as an HTML sanitizer. **BudouX doesn't sanitize any inputs.** Malicious HTML inputs yield malicious HTML outputs. Please use it with an appropriate sanitizer library if you don't trust the input.
