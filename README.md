<!-- markdownlint-disable MD014 -->
# BudouX

[![PyPI](https://img.shields.io/pypi/v/budoux?color=blue)](https://pypi.org/project/budoux/) [![npm](https://img.shields.io/npm/v/budoux?color=yellow)](https://www.npmjs.com/package/budoux)

Standalone. Small. Language-neutral.

BudouX is the successor to [Budou](https://github.com/google/budou), the machine learning powered line break organizer tool.

![Example](https://raw.githubusercontent.com/google/budoux/main/example.png)

It is **standalone**. It works with no dependency on third-party word segmenters such as Google cloud natural language API.

It is **small**. It takes only around 15 KB including its machine learning model. It's reasonable to use it even on the client-side.

It is **language-neutral**. You can train a model for any language by feeding a dataset to BudouX’s training script.

Last but not least, BudouX supports HTML inputs.

## Demo

<https://google.github.io/budoux>

## Natural languages supported by default

- Japanese
- Simplified Chinese

## Supported Programming languages

- Python
- [JavaScript](https://github.com/google/budoux/tree/main/javascript/)

For details about the JavaScript module, please visit [JavaScript README](https://github.com/google/budoux/tree/main/javascript/README.md).

## Python module

### Install

```shellsession
$ pip install budoux
```

### Usage

#### Library

You can get a list of phrases by feeding a sentence to the parser.
The easiest way is to get a parser is loading the default parser for each language.

**Japanese:**

```python
import budoux
parser = budoux.load_default_japanese_parser()
print(parser.parse('今日は天気です。'))
# ['今日は', '天気です。']
```

**Simplified Chinese:**

```python
import budoux
parser = budoux.load_default_simplified_chinese_parser()
print(parser.parse('今天是晴天。'))
# ['今天', '是', '晴天。']
```

You can also translate an HTML string by wrapping phrases with non-breaking markup.

```python
print(parser.translate_html_string('今日は<b>とても天気</b>です。'))
# <span style="word-break: keep-all; overflow-wrap: break-word;">今日は<b ><wbr>とても<wbr>天気</b>です。</span>
```

If you have a custom model, you can use it as follows.

```python
with open('/path/to/your/model.json') as f:
  model = json.load(f)
parser = budoux.Parser(model)
```

A model file for BudouX is a JSON file that contains pairs of a feature and its score extracted by machine learning training.
Each score represents the significance of the feature in determining whether to break the sentence at a specific point.

For more details of the JavaScript model, please refer to [JavaScript module README](https://github.com/google/budoux/tree/main/javascript/README.md).

#### CLI

You can also format inputs on your terminal with `budoux` command.

```shellsession
$ budoux 本日は晴天です。 # default: japanese
本日は
晴天です。

$ budoux -l ja 本日は晴天です。
本日は
晴天です。

$ budoux -l zh-hans 今天天气晴朗。
今天天气
晴朗。
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
usage: budoux [-h] [-H] [-m JSON | -l LANG] [-d STR] [-V] [TXT]

BudouX is the successor to Budou,
the machine learning powered line break organizer tool.

positional arguments:
  TXT                      text (default: None)

options:
  -h, --help               show this help message and exit
  -H, --html               HTML mode (default: False)
  -m JSON, --model JSON    custom model file path (default: /path/to/models/ja-knbc.json)
  -l LANG, --lang LANG     language of custom model (default: None)
  -d STR, --delim STR      output delimiter in TEXT mode (default: ---)
  -V, --version            show program's version number and exit

supported languages of `-l`, `--lang`:
- zh-hans
- ja
```

## Caveat

BudouX supports HTML inputs and outputs HTML strings with markup that wraps phrases, but it's not meant to be used as an HTML sanitizer. **BudouX doesn't sanitize any inputs.** Malicious HTML inputs yield malicious HTML outputs. Please use it with an appropriate sanitizer library if you don't trust the input.

## Background

English text has many clues, like spacing and hyphenation, that enable beautiful and readable line breaks. However, some CJK languages lack these clues, and so are notoriously more difficult to process. Line breaks can occur randomly and usually in the middle of a word or a phrase without a more careful approach. This is a long-standing issue in typography on the Web, which results in a degradation of readability.

Budou was proposed as a solution to this problem in 2016. It automatically translates CJK sentences into HTML with lexical phrases wrapped in non-breaking markup, so as to semantically control line breaks. Budou has solved this problem to some extent, but it still has some problems integrating with modern web production workflow.

The biggest barrier in applying Budou to a website is that it has dependency on third-party word segmenters. Usually a word segmenter is a large program that is infeasible to download for every web page request. It would also be an undesirable option making a request to a cloud-based word segmentation service for every sentence, considering the speed and cost. That’s why we need a standalone line break organizer tool equipped with its own segmentation engine small enough to be bundled in a client-side JavaScript code.

Budou*X* is the successor to Budou, designed to be integrated with your website with no hassle.

## How it works

BudouX uses the [AdaBoost algorithm](https://en.wikipedia.org/wiki/AdaBoost) to segment a sentence into phrases by considering the task as a binary classification problem to predict whether to break or not between all characters. It uses features such as the characters around the break point, their Unicode blocks, and combinations of them to make a prediction. The output machine learning model, which is encoded as a JSON file, stores pairs of the feature and its significance score. The BudouX parser takes a model file to construct a segmenter and translates input sentences into a list of phrases.

## Building a custom model

You can build your own custom model for any language by preparing training data in the target language.
A training dataset is a large text file that consists of sentences separated by phrases with the separator symbol "▁" (U+2581) like below.

```text
私は▁遅刻魔で、▁待ち合わせに▁いつも▁遅刻してしまいます。
メールで▁待ち合わせ▁相手に▁一言、▁「ごめんね」と▁謝れば▁どうにか▁なると▁思っていました。
海外では▁ケータイを▁持っていない。
```

Assuming the text file is saved as `mysource.txt`, you can build your own custom model by running the following commands.

```shellsession
$ pip install .[dev]
$ python scripts/encode_data.py mysource.txt -o encoded_data.txt
$ python scripts/train.py encoded_data.txt -o weights.txt
$ python scripts/build_model.py weights.txt -o mymodel.json
```

Please note that `train.py` takes time to complete depending on your computer resources.
Good news is that the training algorithm is an [anytime algorithm](https://en.wikipedia.org/wiki/Anytime_algorithm), so you can get a weights file even if you interrupt the execution. You can build a valid model file by passing that weights file to `build_model.py` even in such a case.

## Constructing a training dataset from the KNBC corpus for Japanese

The default model for Japanese (`budoux/models/ja_knbc.json`) is built using the [KNBC corpus](https://nlp.ist.i.kyoto-u.ac.jp/kuntt/).
You can create a training dataset, which we name `source_knbc.txt` here, from that corpus by running the command below.

```shellsession
$ python scripts/load_knbc.py -o source_knbc.txt
```

## Author

[Shuhei Iitsuka](https://tushuhei.com)

## Disclaimer

This is not an officially supported Google product.
