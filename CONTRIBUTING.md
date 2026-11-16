# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

### AI-assisted contributions

All submissions to Google Open Source projects need to follow Google's
Contributor License Agreement (CLA), which covers any original work of
authorship included in the submission. This doesn't prohibit the use of coding
assistance tools, including tool-, AI-, or machine-generated code, as long as
these submissions abide by the CLA's requirements.

In addition:

- Non-human tools (such as AI coding assistants) must **not** be listed as
  commit authors or co-authors (for example, do not add `Co-authored-by:`
  trailers for AI tools; disclosure such as `Assisted-by:` is permitted).
- Contributors are fully responsible for understanding, testing, and verifying
  every line they submit against the current `main` branch. Unverified or
  automated cross-port diff-hunting pull requests will be closed.

## Contribution process

### Discuss behavioral or cross-port changes first

- **Bug fixes, typos, and dependency/CI fixes**: You may open a pull request
  directly if it fixes a clear crash, regression, or documented behavior.
- **New features, API changes, or segmentation/HTML behavior changes**: Open an
  issue **before** submitting a pull request, including a concrete real-world
  typesetting use case.
- **Cross-language consistency**: BudouX maintains Python, JavaScript, and Java
  implementations. Behavioral changes to core parsing or HTML processing must
  be agreed upon as a project-wide specification rather than patching a single
  port in isolation to mirror implementation details of another port.

### Code reviews and checks

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Before opening a pull request, run the relevant checks locally:

- **Python**: `uv run ruff format --check .`, `uv run ruff check`,
  `uv run pyrefly check`, and `uv run pytest ./tests`
- **JavaScript**: `npm run lint` and `npm test` (in `javascript/`)
- **Java**: `mvn spotless:check package` (in `java/`)
- **Markdown**: Keep prose lines within 80 characters (`.markdownlint.yaml`)
- **License headers**: New source files must include the Apache 2.0 header with
  `Copyright <YEAR> Google LLC`

## HTML processing scope and non-goals

BudouX is a lightweight phrase segmenter, not a general-purpose HTML parser,
formatter, or sanitizer. Each port uses idiomatic, minimal-dependency HTML
tooling (`DOMParser`/`linkedom` in JS, `html.parser` in Python, `Jsoup` in
Java). Because of this design, we enforce the following boundaries:

1. **Valid HTML5 shorthand (in scope)**: Standard HTML5 void elements (such as
   `<br>` and `<wbr>`) and valid optional end tags (such as `<p>`, `<li>`, and
   `<rt>`) are in scope for consistent phrase segmentation across ports.
2. **Unmatched tags (safety only)**: Missing or stray end tags (such as
   `abc</p>`) must not crash the parser or leak skip state into subsequent
   blocks, but **how unmatched tags are repaired or serialized in output is not
   guaranteed to match across ports**.
3. **Explicit non-goals (pull requests solely targeting these will be closed)**:
   - **Byte-for-byte HTML serialization parity** across ports (for example,
     `<br>` vs. `<br/>`, attribute ordering or quoting, entity encoding choices,
     comment or doctype preservation differences, or `style` attribute spacing).
   - **Complex HTML5 error-recovery algorithms** in non-DOM ports (such as the
     adoption agency algorithm for misnested tags like `<b><nobr>...</b>` or
     table foster parenting).
   - **CSS style computation outside a live browser DOM** (resolving computed
     `display` or `white-space` styles in string-based APIs).
