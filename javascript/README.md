<!-- markdownlint-disable MD014 -->
# BudouX JavaScript module

BudouX is a standalone, small, and language-neutral phrase segmenter tool that provides beautiful and legible line breaks.

For more details about the project, please refer to the [project README](https://github.com/google/budoux/).

## Demo

<https://google.github.io/budoux>

## Install

```shellsession
$ npm install budoux
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

## Web component

BudouX also has a custom element to make it easy to integrate the parser with your website.
All you have to do is wrap sentences with the `<budoux-ja>` tag.

```html
<budoux-ja>今日はとても天気です。</budoux-ja>
```

In order to enable the custom element, you can simply add this line to load the bundle.

```html
<script src="https://unpkg.com/budoux/bundle/budoux-ja.min.js"></script>
```

Otherwise, if you wish to bundle the component with the rest of your source code, you can import the component as shown below.

```javascript
import 'budoux/module/webcomponents/budoux-ja';
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

```shellsession
$ budoux 本日は晴天です。 -H
<span style="word-break: keep-all; overflow-wrap: break-word;">本日は<wbr>晴天です。</span>
```

If you want to see help, run `budoux -h`.

```shellsession
$ budoux -h
Usage: budoux [-h] [-H] [-m JSON] [-d STR] [-V] [TXT]

BudouX is the successor to Budou, the machine learning powered line break organizer tool.

Arguments:
  txt                 text

Options:
  -H, --html          HTML mode
  -d, --delim <str>   output delimiter in TEXT mode (default: "---")
  -m, --model <json>  custom model file path
  -V, --version       output the version number
  -h, --help          display help for command
```

### Attributes

- thres
  - The threshold value to control the granularity of output chunks. Smaller value returns more granular chunks. (default: 1000).

## Caveat

BudouX supports HTML inputs and outputs HTML strings with markup applied to wrap phrases, but it's not meant to be used as an HTML sanitizer. **BudouX doesn't sanitize any inputs.** Malicious HTML inputs yield malicious HTML outputs. Please use it with an appropriate sanitizer library if you don't trust the input.

## Author

[Shuhei Iitsuka](https://tushuhei.com)

## Disclaimer

This is not an officially supported Google product.
