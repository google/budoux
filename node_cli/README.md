# BudouX Node.js CLI

This package is a cli tool for [BudouX](<(../README.md).>)

## Install

```
npm install budoux-cli
```

## Usage

### Simple usage

```bash
$ npm install -g budoux-cli
$ budoux-cli -H '今日は<b>とても天気</b>です。'
# <span style="word-break: keep-all; overflow-wrap: break-word;">今日は<b><wbr>とても<wbr>天気</b>です。</span>

# Or just run temporary to use npx.
$ npx budoux-cli '今日はとても天気です。'
# 今日は
# とても
# 天気です。
```

## Author

[Shuhei Iitsuka](https://tushuhei.com)

## Disclaimer

This is not an officially supported Google product.
