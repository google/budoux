# BudouX Java Module

BudouX is a standalone, small, and language-neutral phrase segmenter tool that
provides beautiful and legible line breaks.

For more details about the project, please refer to the [project README](https://github.com/google/budoux/).

## Demo

<https://google.github.io/budoux>

## Usage

### Simple usage

You can get a list of phrases by feeding a sentence to the parser.
The easiest way is to get a parser is loading the default parser for each language.

```java
import com.google.budoux.Parser;

public class App
{
    public static void main( String[] args )
    {
        Parser parser = Parser.loadDefaultJapaneseParser();
        System.out.println(parser.parse("今日は良い天気ですね。"));
        // [今日は, 良い, 天気ですね。]
    }
}
```

#### Supported languages and their default parsers
- Japanese: `Parser.loadDefaultJapaneseParser()`
- Simplified Chinese: `Parser.loadDefaultSimplifiedChineseParser()`
- Traditional Chinese: `Parser.loadDefaultTraditionalChineseParser()`

### Working with HTML

If you want to use the result in a website, you can use the `translateHTMLString`
method to get an HTML string with non-breaking markup to wrap phrases.

```java
System.out.println(parser.translateHTMLString("今日は<strong>良い天気</strong>ですね。"));
//<span style="word-break: keep-all; overflow-wrap: break-word;">今日は<strong><wbr>良い<wbr>天気</strong>ですね。</span>
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
