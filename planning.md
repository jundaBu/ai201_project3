# Project 3 — TakeMeter Planning

A fine-tuned text classifier that evaluates **discourse quality** in r/soccer. Define the
labels → collect & annotate ≥200 comments → fine-tune a model → compare to a zero-shot
baseline → honestly evaluate where it works and where it breaks.

> **Note:** the training/eval code lives in a **Google Colab notebook**; this repo holds the
> planning, README, and dataset artifacts.

## Required features (deliverables checklist)

- [ ] **Label taxonomy (2–4 labels)** — mutually exclusive, ≥90% coverage with no catch-all
      "other", grounded in community norms. *(designed below)*
- [ ] **Annotated dataset (≥200 examples)** — train/val/test split; README documents source,
      labeling process, label distribution (count per label), and ≥3 genuinely hard cases +
      decisions.
- [ ] **Fine-tuning pipeline** — fine-tune `distilbert-base-uncased`; README states the base
      model, training approach, and ≥1 key hyperparameter decision (LR / epochs / batch size).
- [ ] **Baseline comparison** — zero-shot Groq `llama-3.3-70b-versatile` on the same test set.
- [ ] **Evaluation report** — accuracy for both models, ≥1 per-class metric (P/R/F1),
      confusion matrix, ≥3 wrong predictions analyzed, and a reflection on *what the model
      learned vs. what we intended*.

> **Milestone 2 crosswalk:** (1) Community → *Community*; (2) Labels → *Labels*;
> (3) Hard edge cases → *Hardest edge case + decision rule*; (4) Data collection →
> *Data collection plan*; (5) Evaluation metrics → *Evaluation metrics & why*;
> (6) Definition of success → *Definition of success*. Plus *AI Tool Plan*.

## 1. Community

**r/soccer** — one of the largest football (soccer) communities online (~5M members).
Discourse is active, text-heavy, and ranges widely in quality: from rigorous tactical
breakdowns, to bold opinions stated as fact, to the subreddit's signature **banter**
(jokes, one-liners, flair-based ribbing). Every user wears a club **flair** next to their
name, so most exchanges carry an implicit allegiance.

**Why it's a good fit for classification:** the discourse is genuinely *varied in quality* —
the same match thread contains a stat-driven tactical post and a one-line joke side by side,
so the classes co-occur in the same context rather than being trivially separable by topic.
The quality distinction is also **native to the community**: regulars constantly, informally
judge whether a comment is *actually saying something* or just noise — that judgment is
literally what drives upvotes. That makes the labels real (people already think this way)
and the task non-trivial (the signal is in *how* something is said, not *what* it's about).

## Unit of annotation: **comments**, not post titles

On r/soccer, submissions are mostly news links, transfer reports, and video clips — the
title is just a headline ("Romano: Here we go"). The real discourse lives in **top-level
comments** on match threads, post-match threads, and discussion/opinion posts. So we
annotate **comments**, filtering out pure link/image submission titles.

## 2. Labels (3)

Resolved by a strict **decision tree** (below) so every comment gets exactly one label.

### `analysis`
A structured, evidence-backed claim. Cites something specific and verifiable — a stat, a
tactical observation, a historical comparison. **The reasoning is the point.**

- ✅ "City's build-up collapsed when Rodri went off — the double pivot couldn't progress
  centrally, so they went long 14 times in the 2nd half vs 3 in the 1st. That's why
  Haaland was isolated."
- ✅ "Arsenal's xG is 2.1/game over the last 10 but they're scoring 1.3. That's not a
  creativity problem, it's finishing — the chances are being created."
- ❓ *Uncertain:* "Saka's been their best player all season and the numbers back it up."
  → gestures at evidence ("the numbers") without actually providing any. See decision tree.

### `hot_take`
A bold, confident **evaluative opinion** asserted without real supporting evidence (or
with only decorative/cherry-picked evidence). The author genuinely holds the view; the
claim is *stated*, not *argued*.

- ✅ "Mbappé is the most overrated player in world football. Disappears in every big game."
- ✅ "Ten Hag is single-handedly why United are finished. Sack him today."
- ❓ *Uncertain:* "Bruno is a top-5 midfielder in the world, his stats prove it." → asserts
  boldly *and* name-drops "stats" but provides none. See decision tree.

### `banter`
A comment whose **primary purpose is humor** — jokes, one-liners, memes, flair-based
ribbing, copypasta. Its upvotes come from being funny, not from informing or persuading.
This is the distinctive r/soccer category.

- ✅ "He's got a wand of a left foot and a brick for a right."
- ✅ "Brighton selling another player to a top-6 club for £50m — name a more iconic duo."
- ❓ *Uncertain:* "Imagine paying £85m for Antony to do step-overs into the corner flag 💀"
  → funny (banter) but also asserts Antony is overpriced/bad (hot_take). **This is the
  hardest edge case — see below.**

## Decision tree (enforces mutual exclusivity)

Apply in order; stop at the first match:

1. **Is the primary intent humor?** (the comment's value rests on the joke/delivery)
   → `banter`
2. **Else, does it provide specific, verifiable evidence that carries the claim?**
   (an actual stat, a concrete tactical observation, a real comparison)
   → `analysis`
3. **Else** (a bold opinion with no real evidence, or only decorative evidence)
   → `hot_take`

This ordering resolves every overlap deterministically: a vague appeal to evidence
("the numbers back it up", "his stats prove it") fails step 2 and falls through to
`hot_take`.

## 3. Hardest edge case + decision rule

1.**The funny-but-opinionated comment:** *"Imagine paying £85m for Antony to do step-overs
into the corner flag 💀"*

It could be `banter` (it's a joke, upvoted for being funny) or `hot_take` (it asserts
Antony is bad / overpriced).

**Rule:** Classify by **primary communicative intent**. Ask: *"If this comment weren't
funny, would it still carry its weight as a take?"*
- If the value rests on the joke/delivery → `banter`.
- If it's a claim the author is genuinely advancing and the humor is just seasoning →
  `hot_take`.

The Antony comment is posted and upvoted **for the joke**; the opinion is the vehicle, not
the point → **`banter`**. (Step 1 of the tree formalizes this: humor-first wins.)

A second, softer boundary — `analysis` vs `hot_take` — is handled entirely by step 2:
the evidence must be *present and load-bearing*, not merely gestured at.

2."Conte likes leaving. Also, we may offer him more money.
Between banter and hot_take, final decision is hot_take

3."I swear he's been around 10 years now. If hes 21 then he must have come through after/during COVID which can't possibly be true"
Between hot_take and analysis, final decision hot take

## Mutual-exclusivity check

The decision tree forces exactly one label per comment. The two real overlap zones
(joke+opinion → step 1; opinion+vague-stat → step 2 fails → step 3) both have a
deterministic answer.

## 4. Data collection plan

- **Where:** r/soccer top-level **comments** via the official Reddit API (`praw`); pullpush.io
  (Pushshift successor) as a fallback. *(Reddit blocks plain scraping, so the Colab notebook —
  which can authenticate to the API — is where collection runs.)* Store each comment with its
  permalink.
- **Sample across thread types** so one label doesn't swamp the set: **match threads** skew
  heavily toward `banter`; **post-match threads** and **tactical/discussion posts** yield more
  `analysis` and `hot_take`. Deliberately pull from all three.
- **How many per label (target):** ~**200–240 total**, aiming for a *floor* of **≥50 per
  label**. Natural distribution will skew toward `banter`; I'll accept mild imbalance but cap
  any single class at **~50%** by over-sampling discussion/post-match threads for the rarer
  classes. `analysis` is expected to be the scarcest.
- **If a label is underrepresented after 200** (likely `analysis`): do a **targeted second
  pass** — search analysis-heavy threads (post-match tactical posts, big-match discussion,
  flaired "tactics" content) and pull more until that class clears the ≥50 floor, rather than
  inventing examples or collapsing the label. If it stays unreachable, document it and report
  per-class metrics with the imbalance flagged (don't hide it behind accuracy).
- **Split:** stratified **70/15/15** train/val/test, stratified by label so every class
  appears in each split.
- **Hard cases:** log ≥3 genuinely difficult comments + the label chosen + which decision-tree
  step decided it (required in the README).
- ⚠️ The example comments in §2 are *representative* (written to illustrate each boundary,
  since Reddit couldn't be fetched here). During collection, **read 30–40 real comments first
  and replace each example slot with a real, verbatim comment + permalink** — the
  "read before you label" step.

## Considered & rejected

- **`reaction`** (the assignment's generic example label): real, but on r/soccer it blurs
  heavily into `banter` in match threads and into `hot_take` elsewhere, hurting annotator
  agreement. `banter` is the more distinctive, more separable choice for *this* community.

## Fine-tuning plan

- **Base model:** `distilbert-base-uncased` (HuggingFace `Trainer`).
- **Setup:** add a 3-class classification head; tokenize (max_len ~128 — comments are short);
  `AutoModelForSequenceClassification`.
- **Key hyperparameter decision to document:** start at **LR 2e-5, 3–4 epochs, batch size 16**,
  use the **val set + early stopping** to pick the epoch — and write up *why* (small dataset →
  few epochs to avoid overfitting; report what we saw and what we changed).

## Baseline plan

- **Zero-shot** Groq `llama-3.3-70b-versatile`: prompt it with the label definitions + decision
  tree and ask for one label per **test** comment (no training).
- Run on the **exact same test set** as the fine-tuned model for a fair comparison.

## 5. Evaluation metrics & why

Computed for **both** models on the **same** test set.

- **Macro-F1** = the *primary* metric. **Why accuracy alone is not enough:** the classes are
  imbalanced (`banter` likely ~50%, `analysis` scarce), so a lazy model that just predicts the
  majority class can score high accuracy while being useless on the class we most care about.
  Macro-F1 averages F1 across the three classes *equally*, so the rare `analysis` class can't be
  ignored — it's the honest headline number here.
- **Overall accuracy** — reported, but read against the **majority-class baseline** (e.g. if
  `banter` is 50%, accuracy must clear 50% by a real margin to mean anything).
- **Per-class precision / recall / F1** (`sklearn.classification_report`) — because the use case
  cares unequally about the classes. For a "surface the good takes" filter, **`analysis`
  precision** matters most (a noisy filter is worthless); for a "flag low-effort takes" use,
  `hot_take` recall matters more.
- **Confusion matrix** — to locate *which boundary* fails. Hypothesis: most errors land on the
  `hot_take ↔ banter` and `analysis ↔ hot_take` boundaries (our two hard edges), not random.
- **Error analysis:** pull **≥3 misclassified** examples and explain *why* (e.g., a joke
  containing a real claim read as `hot_take`; a long opinion with no evidence read as
  `analysis`).
- **Reflection:** what the model actually learned (surface cues — emojis, length, the literal
  token "xG", "💀") vs. what we intended (the analysis/hot_take/banter *intent* distinction).

## 6. Definition of success

Specific, objectively checkable thresholds:

- **Floor (project worked at all):** fine-tuned model **beats both** the majority-class baseline
  **and** the Groq zero-shot baseline on **macro-F1**. If fine-tuning doesn't beat zero-shot, it
  added no value — that's a finding, but not success.
- **Target (genuinely good):** **macro-F1 ≥ 0.70** and **overall accuracy ≥ 0.75**.
- **Per-class bar:** **`analysis` precision ≥ 0.70 and recall ≥ 0.60** — the scarce, high-value
  class must be both findable and trustworthy, not sacrificed for overall accuracy.
- **Deployment-useful bar:** usable as a community tool (e.g. a "show me the analysis" filter or
  a low-effort-take flag) **iff `analysis` precision ≥ 0.70** — i.e. when it surfaces a comment
  as analysis, it's right ≥7/10 times, so the filter isn't noise. Below that, it's a demo, not a
  tool.

These are pass/fail at the end: compute macro-F1, the two baselines, and `analysis` P/R, then
check each line.

## Stretch consideration

Macro-F1 0.70 assumes ~50+ examples/class. If `analysis` stays scarce, prioritize its
precision over headline accuracy and say so explicitly in the report.

## AI Tool Plan

There's no app code to generate here, so AI tools help at three specific points. An explicit
decision is made for each (even where I skip).

- **Label stress-testing — YES, do before annotating.** Give an LLM (Claude) the three label
  definitions + the decision tree + the hardest edge case, and ask it to generate **8–10
  comments that sit on the `analysis↔hot_take` and `banter↔hot_take` boundaries**. Run each
  through my decision tree. If any can't be classified cleanly, the definitions are too loose —
  **tighten them now**, before annotating 200 examples. *(This is the highest-leverage AI use in
  the project and can be run immediately.)*
- **Annotation assistance — YES, as drafts in a separate column.** Groq
  `llama-3.3-70b-versatile` writes a **`suggested_label`** draft for each comment (see
  `collect_and_prelabel.py`, cell 5). **I author the real `label` column myself** by reading each
  comment and running the decision tree — deciding from the text first, then glancing at the
  suggestion, never rubber-stamping. The `suggested_label` column stays in the CSV as the
  disclosure record of which rows were AI-assisted. **Test-set integrity:** because the zero-shot
  baseline *is* that same LLM, the test ground truth must be my judgment, not the model's — every
  row is human-decided, and for maximum rigor I label a held-out ~30 rows **blind** (suggestion
  hidden) and keep them as the test set. *(The notebook auto-splits 70/15/15, so this is the
  practical substitute for "no LLM on test.")*
- **Failure analysis — YES.** After evaluation, hand the LLM the **list of misclassified test
  examples (text + true label + predicted label)** and ask it to surface patterns ("does it
  confuse short banter with hot_take?", "does emoji presence drive a class?"). **Verify every
  claimed pattern myself** against the confusion matrix and the actual examples before writing it
  into the report — the AI proposes hypotheses; I confirm them with the data.


