<!-- markdownlint-disable MD014 -->
# BudouX JavaScript module

BudouX is a standalone, small, and language-neutral phrase segmenter tool that
provides beautiful and legible line breaks.

For more details about the project, please refer to the [project README](https://github.com/google/budoux/).

## Demo

<https://google.github.io/budoux>

## Install

```shellsession
$ npm install budoux
```

## Usage

### Simple usage

You can get a list of phrases by feeding a sentence to the parser.
The easiest way is to get a parser is loading the default parser for each language.

**Japanese:**

```javascript
import { loadDefaultJapaneseParser } from 'budoux';
const parser = loadDefaultJapaneseParser();
console.log(parser.parse('今日は天気です。'));
// ['今日は', '天気です。']
```

**Simplified Chinese:**

```javascript
import { loadDefaultSimplifiedChineseParser } from 'budoux';
const parser = loadDefaultSimplifiedChineseParser();
console.log(parser.parse('是今天的天气。'));
// ['是', '今天', '的', '天气。']
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

Internally, the `applyElement` calls the [`HTMLProcessor`] class
with a `<wbr>` element as the separator.
You can use the [`HTMLProcessor`] class directly if desired.
For example:

```javascript
import { HTMLProcessor } from 'budoux';
const ele = document.querySelector('p.budou-this');
const htmlProcessor = new HTMLProcessor(parser, {
  separator: ' '
});
htmlProcessor.applyToElement(ele);
```

[`HTMLProcessor`]: https://github.com/google/budoux/blob/main/javascript/src/html_processor.ts

### Loading a custom model

You can load your own custom model as follows.

```javascript
import { Parser } from 'budoux';
const model = JSON.parse('{"BB2:108120": 1817}');  // Content of the custom model JSON file.
const parser = new Parser(new Map(Object.entries(model)));
```

## Web components

BudouX also offers Web components to integrate the parser with your website quickly.
All you have to do is wrap sentences with `<budoux-ja>` for Japanese,
and `<budoux-zh-hans>` for Simplified Chinese.

```html
<budoux-ja>今日は天気です。</budoux-ja>
<budoux-zh-hans>今天是晴天。</budoux-zh-hans>
```

In order to enable the custom element, you can simply add this line to load the bundle.

```html
<!-- For Japanese -->
<script src="https://unpkg.com/budoux/bundle/budoux-ja.min.js"></script>

<!-- For Simplified Chinese -->
<script src="https://unpkg.com/budoux/bundle/budoux-zh-hans.min.js"></script>
```

Otherwise, if you wish to bundle the component with the rest of your source code,
you can import the component as shown below.

```javascript
// For Japanese
import 'budoux/module/webcomponents/budoux-ja';

// For Simplified Chinese
import 'budoux/module/webcomponents/budoux-zh-hans';
```

### CLI

You can also format inputs on your terminal with `budoux` command.

```shellsession
$ budoux 本日は晴天です。
本日は
晴天です。
```

```shellsession
$ echo $'本日は晴天です。\n明日は曇りでしょう。' | budoux
本日は
晴天です。
---
明日は
曇りでしょう。
```

```shellsession
$ budoux 本日は晴天です。 -H
<span style="word-break: keep-all; overflow-wrap: break-word;">本日は<wbr>晴天です。</span>
```

If you want to see help, run `budoux -h`.

```shellsession
$ budoux -h
Usage: budoux [-h] [-H] [-d STR] [-m JSON] [-V] [TXT]

BudouX is the successor to Budou, the machine learning powered line break organizer tool.

Arguments:
  txt                   text

Options:
  -H, --html            HTML mode (default: false)
  -d, --delim <str>     output delimiter in TEXT mode (default: "---")
  -m, --model <json>    custom model file path
  -V, --version         output the version number
  -h, --help            display help for command
```

## Caveat

BudouX supports HTML inputs and outputs HTML strings with markup applied to wrap
phrases, but it's not meant to be used as an HTML sanitizer.
**BudouX doesn't sanitize any inputs.**
Malicious HTML inputs yield malicious HTML outputs.
Please use it with an appropriate sanitizer library if you don't trust the input.

## Author

[Shuhei Iitsuka](https://tushuhei.com)

## Disclaimer

This is not an officially supported Google product.
