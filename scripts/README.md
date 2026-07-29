# BudouX scripts

This directory is a collection of scripts to generate BudouX model files.
The figure below provides an overview of how each script transforms data files,
from corpora and LLM synthetic sample generation to compiled machine learning models.

```mermaid
flowchart TB
    %% Phase 1: External KNBC Corpus Preparation
    subgraph P1 ["1. KNBC External Corpus Preparation (Git-Ignored / Local Cache)"]
        direction TB
        KNBC_RAW["curl Download<br>KNBC_v1.0_090925_utf8.tar.bz2"] --> KNBC_PREP["prepare_knbc.py"]
        KNBC_PREP --> KNBC_SRC["source_knbc.txt<br>(Full Unsplit Corpus)"]
        
        KNBC_PREP -->|"80% Split (--split-dir, shuffled)"| KNBC_TRAIN["knbc_train.txt"]
        KNBC_PREP -->|"10% Split (--split-dir, shuffled)"| KNBC_VAL["knbc_val.txt"]
        KNBC_PREP -->|"10% Split (--split-dir, shuffled)"| KNBC_TEST["knbc_test.txt<br>(Generalization Benchmark)"]
    end

    %% Phase 2: Agentic Synthetic Sample Generation
    subgraph P2 ["2. Agentic Synthetic Sample Generation (synthesize_samples.py)"]
        direction TB
        CLI_IN["CLI Inputs: --issue=413,468,841<br>or --input='...'"] --> AGENT1["Agent 1: Intent Understanding<br>(extract_intent_target)"]
        AGENT1 --> AGENT2["Agent 2: Example Generator<br>(generate_oversample_candidates)"]
        AGENT2 --> OVERLAY["Deterministic Alignment Utility<br>(align_to_base_parser_splits)"]
        OVERLAY --> AGENT3["Agent 3: Linguistic Expert Polisher<br>(prune_linguistic_anomalies)"]
        AGENT3 --> STAGING_TXT["Staging Storage<br>(staging_raw.txt / Ephemeral Temp Files)"]
    end

    %% Phase 3: Human Review & Dataset Persistence
    subgraph P3 ["3. Human Review & Dataset Persistence (Git-Tracked)"]
        direction TB
        STAGING_TXT --> REVIEW_GATE{"Human Sample Review Gate<br>(Yes / Edit / No)"}
        
        REVIEW_GATE -->|"Yes / Edit: Approved"| SYNTH_STORE["ja/issue_*.txt<br>(Issue-specific Samples)"]
        REVIEW_GATE -->|"Append Representative Case (1 per target)"| REG_TSV["tests/quality/ja.tsv<br>(Canonical Test Suite)"]
        REVIEW_GATE -->|"No: Rejected"| CANCEL1["Synthesis Aborted"]
    end

    %% Phase 4: Data Merging & Training Loop
    subgraph P4 ["4. Data Merging & JAX Training Loop (pipeline_retraining.py)"]
        direction TB
        KNBC_TRAIN --> MERGE_DATA["Data Merging & Weighting<br>(1x base corpus, 100x curated lines)"]
        CURATED_HIST["ja/history.txt<br>(Hand-crafted Historical Corpus)"] --> MERGE_DATA
        SYNTH_STORE --> MERGE_DATA
        
        MERGE_DATA --> COMBINED_TXT["tmp_dir/weighted_train.txt"]
        COMBINED_TXT --> ENCODE_PY["Feature Encoding<br>(run_encode_step / encode_data.py)"]
        ENCODE_PY --> CONFLICT_PY["Conflict Resolution<br>(find_conflicts.py -t 0.8)"]
        CONFLICT_PY --> TRAIN_PY["JAX AdaBoost Training<br>(run_train_step / train.py --iter=200000)"]
        
        TRAIN_PY --> BUILD_PY["Model Compilation<br>(run_build_compact_model_step / build_model.py)"]
    end

    %% Phase 5: Unified QA & Output Artifacts
    subgraph P5 ["5. Unified QA Benchmark & Output Artifacts"]
        direction TB
        BUILD_PY --> TEMP_MODEL["tmp_dir/temp_model.json"]
        TEMP_MODEL --> EVAL_PY["Model Benchmark Evaluation<br>(run_evaluation_report / evaluate_model.py -m -t)"]
        
        REG_TSV -->|"1. Quality & Issue Fix Regression Assertions"| EVAL_PY
        KNBC_TEST -->|"2. Generalization F-score Measurement"| EVAL_PY
        
        EVAL_PY --> REPORT["Raw JSON Metrics (stdout)<br>・accuracy, precision, recall, fscore<br>+ Comparative Holdout Delta Report"]
        REPORT --> FINAL_JSON["budoux/models/ja.json<br>(Updated Production Model)"]
    end

    %% Direct Connections (Validation Feature Encoding)
    KNBC_VAL -->|"run_encode_step"| VAL_ENCODED["tmp_dir/val_encoded.txt"]
    VAL_ENCODED -->|"Validation Loss Monitoring"| TRAIN_PY

    %% Border Styles
    style REVIEW_GATE stroke-width:3px
    style REG_TSV stroke-width:3px
    style FINAL_JSON stroke-width:3px
    style REPORT stroke-width:2px
```

---

## Agentic Training Data Synthesis and Deterministic Retraining Pipelines

When model defects or segmentation issues are reported, resolving them
without hardcoding language-specific rules requires synthesizing targeted
training sentences, auditing them with human review, and performing full
AdaBoost retraining.

Two decoupled meta-pipeline scripts manage this workflow:

- [`pipeline_synthesis.py`](https://github.com/google/budoux/tree/main/scripts/pipeline_synthesis.py):
  Handles synthetic sample generation, interactive human review gates, and
  appending to `ja.tsv`.
- [`pipeline_retraining.py`](https://github.com/google/budoux/tree/main/scripts/pipeline_retraining.py):
  Handles deterministic 80:10:10 KNBC corpus split, separated weighted line
  over-sampling (KNBC base corpus: 1x repetition, curated datasets: 100x
  repetition prior to feature encoding), feature conflict resolution, AdaBoost
  training (200,000 iterations), compact JSON export, and holdout/regression
  comparative reports.

### 1. Automated Meta-Pipeline Commands

To generate synthetic samples and register them into the dataset and quality
suites:

```bash
# Synthesize training samples for reported GitHub issues with auto-accept
python scripts/pipeline_synthesis.py --issue=413,468,841 --lang=ja --auto-accept

# Synthesize training samples directly for target phrase segmentation patterns
python scripts/pipeline_synthesis.py --input="いよいよ/はじまる,もはや" --lang=ja --auto-accept
```

To execute deterministic model retraining and evaluation reporting:

```bash
# Deterministically split KNBC 80:10:10, encode with separated weights, train AdaBoost (200,000 iterations), and evaluate holdout/regression suites
python scripts/pipeline_retraining.py --lang=ja --iter=200000 --feature-thres=2
```

---

### 2. Manual Step-by-Step CLI Execution Flow

If you prefer to execute each stage of the data flow manually, follow the
steps below.

#### Step A: Download & Prepare Baseline Corpus (3-Way Split)

Fetch the raw KNBC corpus and split it into 80% Train, 10% Validation, and 10%
Test datasets using `--split-dir`:

```bash
curl -o knbc.tar.bz2 https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2
tar -xf knbc.tar.bz2
python scripts/prepare_knbc.py KNBC_v1.0_090925_utf8 -o source_knbc.txt --split-dir=tmp/splits
# Outputs:
# - source_knbc.txt           (Full unsplit corpus written to CWD)
# - tmp/splits/knbc_train.txt (80% split for AdaBoost fitting)
# - tmp/splits/knbc_val.txt   (10% split for validation loss tracking)
# - tmp/splits/knbc_test.txt  (10% split for generalization benchmark)
```

#### Step B: Synthesize Samples with LLM Multi-Agent System

Generate candidate training sentences using
[`synthesize_samples.py`](https://github.com/google/budoux/tree/main/scripts/synthesize_samples.py):

```bash
python scripts/synthesize_samples.py --issue=468 --lang=ja --output=staging_468.txt
```

#### Step C: Store Approved Curated Datasets

Save approved candidate sentences under `data/finetuning/ja/issue_468.txt`, and
append exactly one representative test case (the first approved sample for the
target) to root `tests/quality/ja.tsv`.

#### Step D: Merge Training Sources & Encode Features

Combine `KNBC Train` (`tmp/splits/knbc_train.txt`) and all curated datasets
(`data/finetuning/ja/*.txt`) with 100x line repetition for curated bug-fix
samples, then extract $n$-gram features:

```bash
# Combine base training corpus with 100x repeated curated datasets:
for f in data/finetuning/ja/*.txt; do for i in {1..100}; do cat "$f"; done; done > tmp/weighted_train.txt
cat tmp/splits/knbc_train.txt >> tmp/weighted_train.txt
python scripts/encode_data.py tmp/weighted_train.txt -o tmp/encoded.txt
```

#### Step E: Resolve Feature Conflicts

Filter out feature vector label contradictions using 80% majority
thresholding:

```bash
python scripts/find_conflicts.py tmp/encoded.txt -o tmp/cleaned.txt -t 0.8
```

#### Step F: JAX AdaBoost Retraining & Model Building

Encode the validation split (`KNBC Val`) into feature TSV format, train
feature weight scores while monitoring validation loss, then compile the model
JSON:

```bash
python scripts/encode_data.py tmp/splits/knbc_val.txt -o tmp/val_encoded.txt
python scripts/train.py tmp/cleaned.txt --val-data=tmp/val_encoded.txt -o tmp/weights.txt --iter=200000 --feature-thres=2
python scripts/build_model.py tmp/weights.txt -o budoux/models/ja.json
```

#### Step G: Benchmark & Quality Suite Evaluation

Run evaluation benchmark against both `KNBC Test` (generalization F-score) and
`tests/quality/ja.tsv` (canonical quality test suite using required `-m` and
`-t` flags):

```bash
# Evaluate quality regression suite:
python scripts/evaluate_model.py -m budoux/models/ja.json -t tests/quality/ja.tsv

# Evaluate generalization holdout suite:
python scripts/evaluate_model.py -m budoux/models/ja.json -t tmp/splits/knbc_test.txt
```

---

## Dataset Governance & Structure (`data/finetuning/{lang}/`)

Fine-tuning datasets are organized under `data/finetuning/{lang}/` to
guarantee transparency, data lineage, and modular rollback:

- `history.txt`: Hand-crafted historical fine-tuning sentences created by
  maintainers (including merged legacy validation samples).
- `issue_*.txt`: Isolated reviewed datasets for specific GitHub bug fixes.
- Root `tests/quality/{lang}.tsv`: Canonical regression test suite located in
  the repository root. Exactly one representative sample
  (`approved_for_target[0]`) from each fixed issue is automatically appended
  here for permanent protection in future `pytest` runs.

---

## Preparing a source text

A source text file is a collection of human-readable sentences that have been
annotated with segmentation.
This is the format used to describe a labeled dataset, which is the very first
step in the training pipeline.
Typically, the file content should look like below.
Note that you should use `▁` (U+2581) for segmentation, not underscore.

```text
今日は▁良い▁天気ですね。
明日も▁天気でしょう。
昨日は▁気候が▁良かった。
```

There is no set way to segment sentences.
You can segment sentences however you need to for your specific purpose.
BudouX attempts to learn the rules behind segmentation and provides you with a
machine learning model that segments unseen sentences in the same way.

The default [Japanese model](https://github.com/google/budoux/tree/main/budoux/models/ja.json)
is trained on sentences that have been segmented into [phrases (Bunsetsu)](https://en.wikipedia.org/wiki/Japanese_grammar#Sentences,_phrases_and_words).
The default [Simplified Chinese](https://github.com/google/budoux/tree/main/budoux/models/zh-hans.json)
and [Traditional Chinese](https://github.com/google/budoux/tree/main/budoux/models/zh-hant.json)
models are trained on sentences segmented into words.
We picked these segmentations to provide a satisfactory reading experience in
those languages, but you can apply another segmentation method that works best
for your purpose when you build a custom model.

You can make a source data file by hand or running data preparation scripts
(`prepare_*.py`) that extracts segmented sentences from a corpus.
Currently, this directory provides data preparation scripts such as
[`prepare_knbc.py`](https://github.com/google/budoux/tree/main/scripts/prepare_knbc.py)
and [`prepare_wisesight.py`](https://github.com/google/budoux/tree/main/scripts/prepare_wisesight.py).
For Japanese, `prepare_knbc.py` generates a source text file from
[Kyoto University and NTT Blog (KNBC) Corpus](https://nlp.ist.i.kyoto-u.ac.jp/kuntt/),
which segments Japanese sentences by phrase.
When we support other corpora as data sources, we should add the data
preparation scripts for them in this directory with the `prepare_` prefix.

The below snippet shows how you can prepare a source text file (`source_knbc.txt`)
from the KNBC corpus.

```bash
curl -o knbc.tar.bz2 https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2
tar -xf knbc.tar.bz2  # outputs KNBC_v1.0_090925_utf8 directory
python scripts/prepare_knbc.py KNBC_v1.0_090925_utf8 -o source_knbc.txt
```

## Encoding a dataset

The next step is to encode a source text file into a format that can be used for
binary classification.
BudouX segments a sentence by analyzing each character to determine if it should
be the end of a segment or connected to the next character.
Hence, each character in a sentence becomes a data point for the binary
classification model, where a positive example indicates that the character is
the end of a segment and a negative example indicates that the character should
be connected to the next.

The encoding process should also extract *features* from the inputs.
A feature is a generated data from the input, which becomes a good signal for a
machine learning algorithm to make useful inferences based on.
BudouX's machine learning model's goal is to predict if the character is
positive or negative from the features generated from.

The encoding script [`encode_data.py`](https://github.com/google/budoux/tree/main/scripts/encode_data.py)
does this job by outputting an encoded data file from a source text file.
You can output an encoded data file named `encoded.txt` from a source file named
`source.txt` by running:

```bash
python scripts/encode_data.py source.txt -o encoded.txt
```

Currently, this script extracts the following types of features looking at the
surrounding 6 characters around the character of interest.
Let's say we're looking at the `i`-th character in a `sentence` and determining
if it's positive (i.e. the end of a segment) or negative (i.e. not the end of a
segment).
The encoding script extracts the features below to make an inference.

- UW1: `sentence.slice(i-2, 1)`
- UW2: `sentence.slice(i-1, 1)`
- UW3: `sentence.slice(i, 1)`
- UW4: `sentence.slice(i+1, 1)`
- UW5: `sentence.slice(i+2, 1)`
- UW6: `sentence.slice(i+3, 1)`
- BW1: `sentence.slice(i-1, 2)`
- BW2: `sentence.slice(i, 2)`
- BW3: `sentence.slice(i+1, 2)`
- TW1: `sentence.slice(i-2, 3)`
- TW2: `sentence.slice(i-1, 3)`
- TW3: `sentence.slice(i, 3)`
- TW4: `sentence.slice(i+1, 3)`

In other words, they are [$n$-grams](https://en.wikipedia.org/wiki/N-gram)
(unigrams, bigrams, and trigrams specifically) around the character of interest.
This encoding approach is heavily influenced by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/),
a lightweight Japanese word segmenter.
We could include `sentence.slice(i-2, 2)` and `sentence.slice(i+2, 2)`
technically speaking, but we're not following the convention of the segmenter.

An encoded data file generated by the script is a TSV file that typically looks
like below.
Please note that the snippet shown below is only a portion of the data generated
from the previous example, and is for illustrative purposes only.

```text
-1	UW1:良	UW2:い	UW3:天	UW4:気	UW5:で
-1	UW1:日	UW2:も	UW3:天	UW4:気	UW5:で
1	UW1:昨	UW2:日	UW3:は	UW4:気	UW5:候
```

Each line in the file represents a single data point. The first column of each
line should be either 1 or -1, which indicates whether the data point is a
positive example or a negative example. This is the target output $y$ that the
machine learning model must eventually predict. Each line may have an arbitrary
number of following items. These following items represent the features, which
become the input $\mathbf{x}$ for the machine learning model.

Let's see the mathematical notation for an input $\mathbf{x}$ and an output $y$
that derives from this file.
If we look at the example output above, there are 11 unique features in the
second column and beyond.

1. UW1:日
1. UW1:昨
1. UW1:良
1. UW2:い
1. UW2:も
1. UW2:日
1. UW3:は
1. UW3:天
1. UW4:気
1. UW5:で
1. UW5:候

This means that each input can be represented as a 11-dimensional vector of
features i.e. $\mathbf{x} = (x_1, \cdots, x_{11}) \in \{-1, +1\}^{11}$, where
each value becomes $+1$ if the corresponding feature is present in the line or
$-1$ otherwise.
Following this rule, the snippet above can be read as a list of inputs
$\mathbf{x}$ and outputs $y$ as follows:

| $y$  | $(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}, x_{11})$ |
|------|-----------------------------------------------------------------|
| $-1$ | $(-1, -1, +1, +1, -1, -1, -1, +1, +1, +1, -1)$                  |
| $-1$ | $(+1, -1, -1, -1, +1, -1, -1, +1, +1, +1, -1)$                  |
| $+1$ | $(-1, +1, -1, -1, -1, +1, +1, -1, +1, -1, +1)$                  |

The goal of our model training is to design a good function $f$ that predicts
the output from the input, i.e. $y = f(\mathbf{x})$ with generalization ability.

We're intentionally keeping the extracted features as simple as possible to make
the entire library language-neutral and easy to port to other programming
environments.
That being said, you can build a custom parser that is strongly optimized for
your specific needs by adding custom features if you have a good insight around
what type of feature would be beneficial.
Historically, we used to employ features that look to specific [Unicode blocks](https://en.wikipedia.org/wiki/Unicode_block),
but we removed them to make the parsing logic simpler and faster to execute
after we found that their contribution is not significant at <https://github.com/google/budoux/pull/86>.

## Training a model

You can train a BudouX model by passing your encoded data file to [`train.py`](https://github.com/google/budoux/tree/main/scripts/train.py),
and it outputs a weight file, a raw representation of the trained model.

```bash
python scripts/train.py encoded.txt -o weights.txt
```

The encoded data file passed to the positional argument is used as a *training*
dataset, which the training script uses to learn the best parameters for the
model to minimize the error in a greedy manner.
You can also pass another encoded data file as a *validation* dataset with the
optional `--val-data` arg, which is strongly recommended to avoid the machine
learning from overfitting to the training data.
The script reports the metrics over the training and validation data on console
and outputs the trace of them to a log file, which you can specify with the
optional `--log` arg.
You can use these metrics to evaluate if the model is underfitting or
overfitting during the training.

The machine learning algorithm employed behind is [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost),
which calculates how much each feature contributes to the end of a segment.
Each contribution is reprensented by a score, which can be a positive or
negative value; a positive value means positive contribution, while a negative
value means negative contribution.
The longer you run the training (i.e. assigning a larger number of iterations to
the `--iter` arg), the more accurate the output model will be, although it
entails a larger model file size by assigning non-zero contribution scores to
more features.
The contributions scores are the parameters that the machine learning algorithm
can tune, and they are what we call *weights*.

A good nature of this algorithm is that it iteratively updates the weights from
the most important features to the least ones.
The training script appends the weight diffs to the output file, which is
specified by the `-o` / `--output` arg, at a frequency specified by the
`--out-span` arg.
Hence, you can build a model file from the output weight file even if you needed
to interrupt the program before it ends (cf. [Anytime algorithm](https://en.wikipedia.org/wiki/Anytime_algorithm)).

A weight file is a TSV file that saves weight *diffs* as a result of training.
The file content looks like below typically.
This is only showing a part of the weight file generated by the example encoded
data above for illustrative purposes.

```text
UW2:日	1.68
UW1:日	-0.76
UW2:日	0.86
UW3:天	-0.73
UW2:日	0.77
UW1:良	-0.68
```

Please note that the same feature may appear more than once, like the `UW2:日`
in the example above.
It's because they're score diffs the program outputs iteratively throughout the
training process.
The values need to be aggregated by features to get the final weight scores.

Let's see how the weights work in BudouX.
The learned weights should be represented as a weight vector $\mathbf{w}$, which
should have the same length as the input vector (i.e. the number of features).
If we take the example weight file above, the vector should be represented as:

$$
\mathbf{w} = (-0.76, 0, 0, 0, 0, (1.68 + 0.86 + 0.77), 0, -0.73, 0, 0, 0) = (
    -0.76, 0, 0, 0, 0, 3.31, 0, -0.73, 0, 0, 0)
$$

Please note that the weight scores corresponding to the features that don't
appear in the weight file become zero.
Also, as shown in the 6th element in the vector, the values should be summed up
if the same feature appears multiple times in the weight file.

BudouX uses a weight vector for prediction by taking the dot product between the
input vector and the weight vector.
If the dot product's sign is positive, the output $y$ becomes $+1$ (positive),
while it becomes $-1$ (negative) otherwise.
In other words, BudouX does the binary classification with the equation below:

$$y = \text{sgn}(\mathbf{w}^\top \mathbf{x})$$

where $\text{sgn}$ is a sign function that is represented as:

```math
\text{sgn}(x) = \left\{
\begin{array}{ll}
+1 & (x > 0) \\
-1 & (\text{otherwise})
\end{array}
\right.
```

Most computations in `train.py` are written in [JAX](https://github.com/google/jax)
to take advantage of just-in-time compilation and computational accelerators
such as GPU and TPU.
Running the training script over a big dataset may take time on CPUs, so we
recommend to run the script in the environment with accelerators such as [Colab](https://colab.research.google.com/).

### Exporting a model file

A model file for BudouX is a JSON file that saves aggregated weight scores
grouped by their feature types.
Below shows an example model generated from the example weight file presented above.

```json
{
    "UW1":{
        "日":-760,
        "良":-680
    },
    "UW2":{
        "日":3310
    },
    "UW3":{
        "天":-730
    }
}
```

Notice that the features are separated between the colon mark, and the latters
are grouped by the former.
The scores are summed up if there are multiple items that belong to the same group.
Also, the scores are scaled and round to integers.
We apply these conversions to make the output model file smaller and the
inference faster.

The keys in the first layer (e.g. `UW1` and `UW2`) are the feature types that
represent how the corresponding features are extracted from the source data, as
covered by the data encoding section in detail.
The keys in the second layer (e.g. `日` and `天`) are the corresponding feature
values.

You can generate a model file by simply passing the weight file to [`build_model.py`](https://github.com/google/budoux/tree/main/scripts/build_model.py).

```bash
python scripts/build_model.py weights.txt -o model.json
```

### Translating a model

JSON is the primary format for the BudouX models, but some libraries (e.g. [ICU](https://icu.unicode.org/))
may want to use another serialization format to store  model data.
You can translate a model JSON file to another format with [`translate_model.py`](https://github.com/google/budoux/tree/main/scripts/translate_model.py)
if it's more useful for your specific purpose.
For example, you can convert a model file to an ICU Resource Bundle by running:

```bash
python scripts/translate_model.py model.json --format=icu > icu_bundle.txt
```

This script is where to add code when we need to support other formats.

## Remediating and updating model weights

When a model file misclassifies character transition boundaries in specific edge
cases (for example, missing phrase segmentations across newly observed character
combinations), resolving the discrepancy requires performing full AdaBoost
linear retraining using `train.py` across updated feature records
(`encoded.txt`).

Historically, partial gradient-descent fine-tuning over established base JSON
models (`finetune.py`) was supported. However, because partial fine-tuning
optimizes existing $n$-gram weights strictly across pre-established vocabulary
keys, it cannot register or weight out-of-vocabulary character transition
features across newly triaged phrases.

Therefore, full AdaBoost model optimization via `train.py` paired with
statistical conflict resolution (`find_conflicts.py -t 0.8`) and model building
(`build_model.py`) is the sole canonical methodology for updating and tuning
BudouX segmentation models.
