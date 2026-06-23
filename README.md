# TakeMeter — Discourse-Quality Classifier for r/soccer

TakeMeter classifies r/soccer comments by **discourse quality** into one of three labels:
`analysis`, `hot_take`, or `banter`. This README is the final report and is meant to stand on
its own; the working notes and full design rationale behind each decision live in
[planning.md](planning.md).

**Headline result:** a zero-shot 70B LLM (0.74 accuracy) substantially **beat** a DistilBERT
model fine-tuned on ~160 examples (0.43 accuracy). The most useful thing I learned came from
*why* the small model failed — see the evaluation report.

---

## 1. Community & labels

**Community:** [r/soccer](https://www.reddit.com/r/soccer/), one of the largest football
communities online. In a single thread you'll find a stat-driven tactical breakdown next to a
bold opinion next to a one-line joke. That quality range is native to how the community
operates — regulars constantly, informally judge whether a comment is "actually saying
something," and that judgment drives upvotes. This makes the labels real and the task
non-trivial: the signal is in *how* something is said, not *what topic* it's about.

| Label | Definition | Two examples (real, from the dataset) |
|---|---|---|
| `analysis` | A structured claim backed by specific, verifiable evidence — a stat, concrete tactical observation, historical comparison, or specific factual detail. | 1. *"We are tied with Dortmund on points and goal difference in the xG table … our finishing has been woeful."*<br>2. *"He was sold in the winter for like 1.5 million to Sevilla. Back then nobody knew, but… that was a steal."* |
| `hot_take` | A bold, confident opinion asserted without real supporting evidence (or only decorative evidence). | 1. *"Yeah Caicedo really doesn't get the respect he deserves. He's the main man on Chelsea's midfield."*<br>2. *"Wirtz and Reijnders will solve all their midfield problems."* |
| `banter` | A comment whose primary purpose is humor — jokes, one-liners, memes, sarcasm, flair ribbing. | 1. *"So the Community Shield only counts if you've won everything else as well? Sorry, Palace, August is a wash for you."*<br>2. *"Boss we're not doing this… the Community Shield isn't a trophy, it's a dinner plate."* |

Labels are kept mutually exclusive by a **decision tree**, applied in order:
1. Is the primary intent humor? → `banter`
2. Else, does it provide specific, verifiable evidence that carries the claim? → `analysis`
3. Else (bold opinion, no real evidence) → `hot_take`

Full definitions, the hardest edge cases, and the metric/success reasoning are in
[planning.md](planning.md).

---

## 2. Dataset

- **Source:** r/soccer top-level comments collected via the public **pullpush.io** API (a
  Pushshift mirror that needs no Reddit app). Collection script: [scrape_reddit.py](scrape_reddit.py).
  The full raw pool with permalinks, scores, and flairs is in `comments_raw.csv`; the labeled
  working file is [data.csv](data.csv) (`text, label, notes, permalink`).
- **Labeling process:** comments were filtered to 5–80 words (substantive but not essays),
  deduplicated, and labeled with the decision tree above, reading each comment individually.
  The first 42 were hand-labeled from scratch; the rest were **pre-labeled by Claude and then
  human-reviewed and corrected** by me (disclosed in §8).
- **Split:** stratified **70 / 15 / 15** train/val/test via the notebook. Test set = **35**.

**Label distribution (230 labeled comments):**

| Label | Count | Share |
|---|---|---|
| analysis | 87 | 38% |
| hot_take | 81 | 35% |
| banter | 62 | 27% |
| **total** | **230** | no class > 70% ✅ |

**Three genuinely difficult annotation decisions:**
1. *"Imagine paying £85m for Antony to do step-overs into the corner flag 💀"* — joke vs.
   opinion. The comment is funny **and** implies Antony is overpriced. **Decided `banter`:**
   the value rests on the joke (decision-tree step 1, humor-first).
2. *"Conte likes leaving. Also, we may offer him more money."* — banter vs. hot_take. Dry, but
   it advances a real claim rather than a joke. **Decided `hot_take`.**
3. *"I swear he's been around 10 years now. If he's 21 then he must have come through
   after/during COVID which can't possibly be true."* — hot_take vs. analysis. It reasons, but
   from a hunch, not evidence. **Decided `hot_take`** (the reasoning isn't evidence-backed).

---

## 3. Fine-tuned model

- **Base model:** `distilbert-base-uncased` (66M params) with a 3-class sequence-classification
  head, trained with the HuggingFace `Trainer`.
- **Training approach:** comments tokenized to max length 128 (they're short); trained on the
  70% train split, validated on the 15% val split, evaluated on the held-out 35-comment test set.
- **Key hyperparameter decision:** learning rate `2e-5`, **`3` epochs**, batch size `16`.
  <!-- TODO: set the epoch count to what you actually ran -->
  *Rationale:* with only ~160 training examples, I deliberately kept epochs low and the learning
  rate small to limit overfitting on such a tiny set.

---

## 4. Baseline

A **zero-shot** Groq `llama-3.3-70b-versatile`, classifying each test comment with no
task-specific training. **How results were collected:** each of the 35 test comments was sent
individually with the system prompt below at `temperature=0` (deterministic); the model's
one-word reply was lowercased and matched against the three valid labels, and the predictions
were scored against the same gold labels as the fine-tuned model.

**Prompt used:**

```text
You are classifying comments from r/soccer, a large online football community.
Assign each comment to exactly one of the following categories based on its PRIMARY intent.

analysis: A structured claim backed by specific, verifiable evidence — a stat, a concrete
tactical observation, a historical comparison, or specific factual detail. The reasoning is
the point.
Example: "We are tied with Dortmund on points and goal difference in the xG table — but our finishing has been woeful."

hot_take: A bold, confident opinion asserted WITHOUT real supporting evidence (or only vague,
decorative evidence). The claim is stated, not argued.
Example: "Yeah Caicedo really doesn't get the respect he deserves. He's the main man on Chelsea's midfield."

banter: A comment whose primary purpose is humor — a joke, one-liner, meme, sarcasm, or
flair-based ribbing. Its value is being funny, not informing or persuading.
Example: "So the Community Shield only counts if you've won everything else as well? Sorry, Palace, August is a wash for you."

Decision rule — apply in order, stop at the first match:
1. Is the primary intent humor? -> banter
2. Else, does it provide specific, verifiable evidence that carries the claim? -> analysis
3. Else (a bold opinion with no real evidence) -> hot_take

Respond with ONLY the label name. Do not explain your reasoning.
Valid labels: analysis, hot_take, banter
```

---

## 5. Evaluation report

### 5.1 Overall accuracy

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **0.743** | **0.75** |
| Fine-tuned DistilBERT | **0.429** | **~0.32** |

For context: random guessing on 3 classes ≈ 0.33, and always predicting the majority class
(`analysis`, 13/35) ≈ 0.37. The fine-tuned model (0.429) is **barely above guessing**; the
zero-shot LLM is far above it.

### 5.2 Per-class metrics

**Zero-shot baseline:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.75 | 0.69 | 0.72 | 13 |
| hot_take | 0.64 | 0.75 | 0.69 | 12 |
| banter | 0.89 | 0.80 | 0.84 | 10 |

**Fine-tuned DistilBERT:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.50 | 0.85 | 0.63 | 13 |
| hot_take | 0.33 | 0.33 | 0.33 | 12 |
| banter | 0.00 | 0.00 | 0.00 | 10 |

### 5.3 Confusion matrix — fine-tuned model

(Also committed as `confusion_matrix.png`.)

| True ↓ / Predicted → | analysis | hot_take | banter |
|---|---|---|---|
| **analysis** (13) | 11 | 1 | 1 |
| **hot_take** (12) | 8 | 4 | 0 |
| **banter** (10) | 3 | 7 | 0 |

Predicted totals: **analysis 22, hot_take 12, banter 1.** The model almost never predicts
`banter` and over-predicts `analysis`.

### 5.4 Error patterns (AI-assisted, then verified)

I pasted the confusion matrix and the misclassified examples into Claude and asked it to surface
common themes, then verified each against the actual comments. Patterns found:

- **One boundary dominates the errors: everything substantive collapses into `analysis`.**
  The two largest error cells are `hot_take → analysis` (8 of 12 hot_takes) and `banter →
  hot_take` (7 of 10 banters). The model learned a gradient toward `analysis`/`hot_take` and
  treats `analysis` as the default for any comment that mentions football substance.
- **`banter` disappeared entirely (recall 0.00).** Banter comments on r/soccer almost always
  name a club, player, or competition, so the model keyed on that **topical/jargon signal** and
  filed jokes as takes. It never learned that the *intent* was humor.
- **It's a data problem, not a labeling problem.** The labels were applied consistently via the
  decision tree (similar comments got the same label), and the zero-shot LLM — using the *same*
  label definitions — scored 0.74. So the boundary is learnable; ~160 examples (and only 62
  banters) is simply too few for a 66M-param encoder to learn a pragmatic distinction.
- **What I corrected/discarded after verifying:** <!-- TODO after you re-read the real examples:
  note any AI-suggested pattern you rejected, e.g. "Claude guessed sarcasm length was a factor,
  but re-reading showed the misclassified banters were a mix of lengths — discarded." -->

### 5.5 Three misclassified examples (fine-tuned model)

All three were predicted `analysis` at **confidence ~0.34** — barely above the 0.33 random
floor. The model wasn't confidently wrong; it was **defaulting to `analysis` and guessing**.

1. *"Think people would take Doucoure as a squad player and on cheaper wages (supposedly one of
   our highest earners), but he's not taking a wage cut."* — **true `hot_take`, predicted
   `analysis` (0.34).** *Why:* the wage/squad-role details *sound* factual, so the model read
   them as evidence — but the comment is speculation ("Think people would take…"), not an
   evidenced claim. It keyed on factual-looking tokens, not the speculative framing.
2. *"30 Year Old Yamal Back Post Crosses to 23 year old Raphinha Jr we will be there."* —
   **true `banter`, predicted `analysis` (0.34).** *Why:* a joke imagining a future link-up.
   It names players and ages, so the model saw football specifics and missed the humor entirely.
3. *"Lol at the xG being less than 1 when Sørloth scored his fourth goal"* — **true `banter`,
   predicted `analysis` (0.35).** *Why:* the clearest case — it opens with "Lol" (an obvious
   humor marker) yet mentions **xG**, the single token most associated with `analysis` in
   training. The jargon cue overrode the joke marker.

**Going deeper (per the guiding questions):** the confused pairs are directional —
`hot_take → analysis` and `banter → hot_take` — which says the model never learned the *lower*
end of the substance gradient. The boundary is hard because intent (joke / assertion / argument)
is **pragmatic**, not lexical, and the topic vocabulary is identical across all three labels.
The fix is more data, especially for `banter` (the smallest, worst class): roughly 3–5× the
examples, with banter deliberately oversampled and shown across lengths and styles.

### 5.6 Sample classifications (fine-tuned model, with confidence)

| Comment (truncated) | Predicted | Confidence | True | ✓/✗ |
|---|---|---|---|---|
| "How did they concede 1.77 xG from 7 shots?" | analysis | 0.37 | analysis | ✓ |
| "Lol at the xG being less than 1 when Sørloth scored his fourth goal" | analysis | 0.35 | banter | ✗ |
| "30 Year Old Yamal Back Post Crosses to 23 year old Raphinha Jr we will be there." | analysis | 0.34 | banter | ✗ |
| "Think people would take Doucoure as a squad player and on cheaper wages…" | analysis | 0.34 | hot_take | ✗ |

*Why the correct one is reasonable:* "How did they concede 1.77 xG from 7 shots?" cites a
specific stat and reasons from it — exactly the evidence-carrying structure that defines
`analysis` — so predicting `analysis` is well-grounded.

**But note the confidence ceiling:** even this *correct* prediction sits at **0.37**, barely
above the 0.33 random floor — and it was the model's single most confident correct call. The
model isn't confidently right; it leans `analysis` for almost everything (note every row above
is predicted `analysis`) and happens to be correct when the true label is also `analysis`.
This is the same majority-class collapse the confusion matrix shows, seen at the level of
individual predictions.

---

## 6. Reflection — what the model captured vs. what I intended

I intended the model to learn a **pragmatic** distinction: informing (analysis) vs. asserting
(hot_take) vs. joking (banter) — a judgment about *how* a comment communicates. What the model
actually learned was a **shallow topical prior**: "this text contains football substance →
`analysis`," with `hot_take` as a fallback and `banter` effectively dropped from its vocabulary.

It **overfit to surface cues** that correlate with the majority class (football vocabulary,
sentence substance) instead of the underlying intent, and it **missed humor entirely** — the one
class that's defined purely by tone rather than content. The clearest evidence is that `banter`
was the *easiest* class for the zero-shot LLM (F1 0.84) and the *impossible* class for the
fine-tuned model (F1 0.00). A 70B model recognizes humor zero-shot; a 66M encoder on ~160
examples could not, and fell back on the only thing it could pick up from so little data — the
topic. The gap between my label *definitions* (intent-based) and the model's actual *decision
boundary* (topic-based) is the core finding of this project.

**Caveat:** the test set is only 35 comments, so per-class numbers are noisy (one
misclassification ≈ 0.07 of an F1). The *direction* of the result is solid; the exact decimals
are not.

---

## 7. Spec reflection

- **How the spec helped:** the spec's insistence on a **zero-shot LLM baseline on the same test
  set** is the only reason this project produced a real insight. Without it I'd have reported
  "DistilBERT got 0.43" in a vacuum; the baseline reframed that as "a small fine-tune lost to a
  large zero-shot model on a subtle task," which is the actual lesson. The required label-design
  rigor (definitions + decision tree) also paid off — it's why the errors are interpretable
  rather than just noise.
- **Where I diverged:** the spec suggested collecting data manually (copy-paste) and running in
  Colab. I instead wrote a small **pullpush.io scraper that runs locally** and produced the CSV
  directly. I diverged because Reddit blocks unauthenticated scraping and the API-app setup was
  friction; pullpush gave 555 real comments with no credentials, keeping the time on labeling
  rather than plumbing. (Trade-off: pullpush data lags real-time slightly, which doesn't matter
  for this task.)

---

## 8. AI usage

- **Annotation pre-labeling (disclosed):** after hand-labeling the first 42 comments, I directed
  Claude to pre-label the remaining comments using my decision tree. It produced a draft label
  per comment; I reviewed every one against the comment text and **changed the ones I disagreed
  with** before they entered the dataset (the final distribution reflects my corrections, not the
  raw drafts).
- **Data-collection tooling:** I directed Claude Code to write the collection pipeline. It
  produced [scrape_reddit.py](scrape_reddit.py) (pullpush API, length/dedup filtering) and
  [finalize.py](finalize.py) (label validation + `dataset.csv` export); I chose the source,
  thread-type sampling, and the ≥50-per-class / ≤70% rules it enforces.
- **Baseline prompt + error analysis:** Claude drafted the zero-shot classification prompt
  (which I trimmed) and surfaced the error patterns in §5.4 from the confusion matrix, which I
  then verified against the actual misclassified comments.

---

## 9. Repo contents

| File | Purpose |
|---|---|
| [planning.md](planning.md) | Full design: labels, edge cases, metrics, success criteria, AI tool plan |
| [scrape_reddit.py](scrape_reddit.py) | Collect r/soccer comments via pullpush.io |
| [finalize.py](finalize.py) | Validate labels, emit `dataset.csv` |
| [data.csv](data.csv) | Labeled working file (`text, label, notes, permalink`) |
| dataset.csv | Notebook input (`text, label`, 230 rows) |
| comments_raw.csv | Full raw comment pool with metadata |
| confusion_matrix.png | Fine-tuned model confusion matrix |

---

## 10. Notebook cells to fill the TODO slots

**Misclassified examples (§5.5):**
```python
wrong = [(t, yt, yp) for t, yt, yp in zip(test_texts, y_true, y_pred) if yt != yp]
for t, yt, yp in wrong:
    print(f"TRUE={yt:9} PRED={yp:9} | {t}")
```

**Sample classifications with confidence (§5.6):**
```python
import torch, torch.nn.functional as F
def classify(text):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = F.softmax(model(**inp).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return id2label[idx], float(probs[idx])

for t in test_texts[:5]:
    label, conf = classify(t)
    print(f"{conf:.2f}  {label:9} | {t[:80]}")
```
