# When Metrics Mistake Fidelity for Quality: Evaluating Machine Translation Metrics on Literary Translation

## 1. Introduction

Automatic machine translation (MT) evaluation metrics — COMET, COMETKiwi, XCOMET — have become the de facto standard for assessing translation quality. Trained on human judgments of technical and news-domain translations, these neural metrics consistently outperform surface-level metrics like BLEU on standard benchmarks. But translation is not a monolithic task: literary translation, with its emphasis on voice, cultural adaptation, and creative interpretation, occupies a fundamentally different space from the technical/news domains these models were trained on.

We investigate this gap using a unique corpus: **seven human Ukrainian translations of George Orwell's *Animal Farm*** spanning 74 years (1947–2021), plus **three AI translation systems**: GPT-5.2 (general-purpose LLM), DeepL (commercial NMT), and LaPa (`lapa-llm/lapa-v0.1.2-instruct`, an LLM fine-tuned specifically for Ukrainian literary translation). By combining automatic metrics with independent computational analyses and a proposed human evaluation, we test whether current MT metrics are systematically biased against the qualities that make literary translation valuable.

**Thesis.** Our computational evidence shows that all three AI systems score at or near the top of every automatic metric while simultaneously exhibiting lower lexical diversity, narrower function-word vocabularies, fewer Ukrainian discourse particles and diminutive forms, and punctuation profiles that mirror English rather than Ukrainian literary norms. Cross-lingual embeddings reveal that the three AI translations are near-identical to each other (mean pairwise similarity 0.936) while human translations genuinely diverge (0.684). We argue that neural MT metrics enforce a normative band of translational acceptability — rewarding source-closeness, surface fluency, and uniformity — while failing to capture the expressiveness, cultural adaptation, and stylistic identity that distinguish skilled literary translation. The inclusion of three architecturally distinct AI systems (general LLM, commercial NMT, domain-tuned LLM) strengthens the claim: the pattern is not an artifact of one model but a property of AI translation itself. We propose a controlled human evaluation to test whether expert literary judgment diverges from these metric rankings in the specific ways our computational analyses predict.

### 1.1 Corpus

| ID | Year | System | Ukrainian Title |
|----|------|--------|----------------|
| T1 | 1947 | Ivan Cherniatynskyi | Колгосп тварин |
| T2 | 1984 | Iryna Dybko* | Хутір тварин |
| T3 | 1991 | Oleksii Drozdovskyi | Скотоферма |
| T4 | 1991 | Yurii Shevchuk | Ферма рай для тварин |
| T5 | 1992 | Natalia Okolitenko | Скотохутір |
| T6 | 2020 | Bohdana Nosenok | Колгосп тварин |
| T7 | 2021 | Viacheslav Stelmakh | Колгосп тварин |
| T8 | — | LaPa LLM (v0.1.2-instruct) | — |
| T9 | — | GPT-5.2 | — |
| T10 | — | DeepL | — |

*Dybko (1984) is a rephrasing/adaptation rather than a direct translation. We retain it as a sanity check but exclude it from ranked comparisons, leaving 6 human translations + 3 AI systems in all rankings.

The corpus comprises 1,367 aligned segments. The three AI systems represent distinct architectures: GPT-5.2 is a general-purpose LLM prompted for literary translation; DeepL is a commercial neural MT system; LaPa is an LLM fine-tuned on Ukrainian literary parallel data. If all three exhibit the same artifacts, the finding generalizes beyond any single model.

---

## 2. Hypotheses

### H1: Metric-Side Problems

**H1a.** *MT metrics reward semantic fidelity over translation quality.* Metrics trained on parallel corpora conflate closeness to the source with quality, penalizing the creative departures that characterize skilled literary translation.

**H1b.** *MT metrics exhibit temporal bias.* Models trained predominantly on modern text systematically underrate translations with archaic vocabulary, older orthographic conventions, or period-appropriate register.

**H1c.** *MT metrics penalize cultural adaptation.* Translations that localize names, idioms, or cultural references — a legitimate and valued literary strategy — receive lower scores because they diverge from the source semantics.

### H2: AI-Side Artifacts

**H2a.** *AI translations are semantically faithful but lexically impoverished.* AI systems produce correct, source-close translations with lower vocabulary diversity than human translators, resulting in text that reads as "translated" rather than native.

**H2b.** *AI translations lack Ukrainian stylistic identity.* AI systems' punctuation, discourse particles, diminutive morphology, and register mirror the English source rather than conforming to Ukrainian literary norms.

**H2c.** *AI translations exhibit artificial consistency.* AI systems produce more uniform output (lower variance in sentence length, style, metric scores) than human translators, whose output naturally varies with content and mood.

**H2d.** *AI translations converge on a single output.* Despite different architectures and training data, AI systems produce near-identical translations — suggesting they optimize for the same objective and arrive at the same "consensus average" rather than producing individual interpretive voices.

### H3: Structural/Methodological Problems

**H3a.** *Metrics and AI systems share a training-data bubble.* Both are trained on large-scale parallel corpora, causing them to agree on what constitutes "good" translation — and this consensus reflects the properties of their training data, not human literary judgment.

**H3b.** *Reference-free metrics cannot distinguish translation strategies.* Without access to the source, these metrics default to measuring fluency and typicality, which advantages conventional translations over creative ones.

### Competing Hypothesis

**H0.** *Metrics capture literary quality, not just fidelity.* Neural metrics trained on human judgments have learned a quality signal that generalizes to literary translation. Low-scoring translations are genuinely weaker; AI systems score high because they are genuinely good. Deviations from metric rankings reflect evaluator noise, not a systematic measurement gap.

If the human evaluation confirms H0, this would suggest literary quality correlates more tightly with semantic fidelity than translation studies commonly assumes. That outcome would be equally publishable and would challenge the field from the opposite direction.

### Falsifiability Criteria

We commit to the following thresholds before data collection:

| If this result occurs... | ...we conclude: |
|---|---|
| TrueSkill human ranking correlates with metric ranking at Spearman ρ ≥ 0.7 | Metrics capture literary preference — H1a is refuted |
| All 3 AI systems' TrueSkill μ places them in the top 4 (matching metric ranking) | Metrics capture literary quality — H2a–H2c weakened |
| LLM-judge fluency ranking correlates with human preference at ρ ≥ 0.8 | No divergence between human and LLM quality judgments — H3a weakened |
| Human rankings of translations scoring below AI preserve the metric ordering (Kendall's τ ≥ 0.8 with metric ranks) | Metrics rank low-scoring translations correctly — H1c weakened |
| AI–AI tie rate in human preference is < 40% | Humans perceive meaningful differences between AI systems — H2d weakened |

---

## 3. Computational Evidence

### 3.1 Automatic MT Metrics

We evaluated all 10 translations using two reference-free neural metrics (COMETKiwi-22, COMETKiwi-XL) on all systems, ranked by COMETKiwi-22:

| Rank | System | COMETKiwi-22 | COMETKiwi-XL |
|------|--------|-------------|-------------|
| 1 | LaPa (LLM) | 0.820 | 0.717 |
| 2 | GPT-5.2 (LLM) | 0.812 | 0.711 |
| 3 | DeepL (LLM) | 0.805 | 0.697 |
| 4 | Stelmakh 2021 | 0.775 | 0.651 |
| 5 | Shevchuk 1991 | 0.738 | 0.580 |
| 6 | Cherniatynskyi 1947 | 0.738 | 0.574 |
| 7 | Nosenok 2020 | 0.727 | 0.577 |
| 8 | Drozdovskyi 1991 | 0.693 | 0.553 |
| 9 | Okolitenko 1992 | 0.672 | 0.500 |
| 10 | Dybko 1984* | 0.541 | 0.307 |

*Sanity check only — excluded from rankings.

`[reference_free_cometkiwi-22_all_systems.png]`

`[reference_free_cometkiwi-xl_distributions_violin_original.png]`

**Key observations:**

1. **All three AI systems occupy the top 3 positions** on both metrics. The ranking is identical across COMETKiwi-22 and COMETKiwi-XL (Spearman ρ = 0.964, p<.001). LaPa — fine-tuned for Ukrainian literary translation — tops the ranking, but the margin over GPT-5.2 (a general-purpose LLM) is only 0.008, confirming that domain tuning has minimal impact on metric scores.

2. **AI score distributions are the narrowest (thinnest violins).** All three AI systems show tighter segment-level score distributions than any human translator. The AI violins are tall and thin; human violins are shorter and wider. AI systems almost never produce a low-scoring segment — they never take risks.

3. **Round-robin COMET-22** (each translation scored using every other as reference) confirms the same hierarchy: GPT-5.2 (0.809) and DeepL (0.803) top the hypothesis ranking; GPT–DeepL is the highest-scoring pair in the entire matrix (0.871/0.876).

`[round_robin_comet-22_all_systems.png]`

### 3.2 Semantic Similarity and AI Convergence

We measured pairwise cosine similarity between all translations using LaBSE (cross-lingual embeddings, all 1,367 segments):

| Pair type | Avg similarity | N pairs |
|-----------|---------------|---------|
| **AI–AI** | **0.936** | 3 |
| Human–AI | 0.765 | 21 |
| Human–Human | 0.684 | 21 |

The convergence gap is **+0.252**: AI systems are 25 points more similar to each other than humans are to each other. The three AI–AI pairs: LaPa–DeepL 0.951, GPT–DeepL 0.933, LaPa–GPT 0.925. These are essentially the same translation with minor surface variation.

*Even excluding Dybko (a free cultural adaptation): H–H rises to 0.739 — the gap remains +0.197.*

**Proximity to English source** (LaBSE cosine similarity, all segments):

| System | Similarity to source |
|--------|---------------------|
| LaPa (LLM) | 0.853 |
| GPT-5.2 (LLM) | 0.836 |
| DeepL (LLM) | 0.848 |
| Cherniatynskyi 1947 (closest human) | 0.780 |
| Stelmakh 2021 | 0.775 |
| Shevchuk 1991 | 0.751 |
| Nosenok 2020 | 0.713 |
| Drozdovskyi 1991 | 0.683 |
| Okolitenko 1992 | 0.598 |
| Dybko 1984 (cultural adapter) | 0.484 |

All AI systems sit above 0.83 — a clear tier above even the closest human translator (0.780). The heatmap visually shows the hot AI block: three systems converging on the same output while humans spread across a much wider range.

`[round_robin_labse_heatmap.png]`

We cross-validated with OpenAI text-embedding-3-small (general-purpose, outside the "translation bubble") on 100 high-variance segments. The direction holds: AI systems are closest to the source and to each other on both models. LaBSE inflates AI's lead compared to OpenAI, confirming that translation-trained embeddings amplify but do not fabricate the pattern.

### 3.3 Source Literalness

| Measure | AI range | Human range (excl. Dybko) | AI rank |
|---------|----------|--------------------------|---------|
| chrF vs English source | 0.464–0.590 | 0.366–0.416 | **Top 3** |
| LaBSE similarity vs source | 0.836–0.853 | 0.598–0.780 | **Top 3** |

GPT-5.2's chrF (0.590) is 42% closer to the English source than the nearest human (Stelmakh, 0.416). LaPa (0.574) and DeepL (0.464) follow. All three AI systems are more source-literal than any human translator by every measure.

### 3.4 Lexical Diversity

| Metric | LaPa | GPT-5.2 | DeepL | Human range (excl. Dybko) | Tool |
|--------|------|---------|-------|--------------------------|------|
| MTLD | 303 | 328 | 301 | 351–413 | lexicalrichness |
| MATTR | 0.843 | 0.849 | 0.840 | 0.853–0.868 | lexicalrichness |
| Hapax ratio | 0.176 | 0.195 | 0.177 | 0.189–0.233 | pymorphy3 |

Every lexical diversity metric tells the same story: AI systems fall below the human range. MTLD averages ~311 for AI vs. ~377 for humans (excl. Dybko) — an 18% gap. GPT-5.2 is the least impoverished of the three AI systems but still below every human translator on MTLD.

### 3.5 Ukrainian Expressiveness

This section measures culturally-grounded features that define Ukrainian literary prose style. All features are computed with `pymorphy3` (Ukrainian morphological analyzer) and `tokenize-uk`. Human averages below exclude Dybko (n=6).

#### 3.5.1 Punctuation and Rhythm

| System | Em dashes /1k | Dash-to-comma ratio |
|--------|--------------|-------------------|
| Humans (avg) | 16.3 | 0.137 |
| LaPa | **3.5** | **0.028** |
| GPT-5.2 | 11.3 | 0.093 |
| DeepL | 9.4 | 0.075 |

Em dashes are a hallmark of Ukrainian literary prose. LaPa uses 3.5 per thousand words — closer to the English original (2.7) than to any human translator (9.7–20.9). GPT-5.2 and DeepL are closer to humans but still underuse dashes. The pattern is consistent: AI systems import English punctuation norms into Ukrainian text.

`[expressiveness_dashboard.png]`

#### 3.5.2 Discourse Particles

Ukrainian discourse particles (ж, таки, ось, бо, аж, ну, мов, наче, etc.) signal emphasis, surprise, hedging, and speaker attitude. They are pragmatic markers with no direct English equivalents.

| System | Particles /1k | Unique types | Top particle (% of total) |
|--------|--------------|-------------|--------------------------|
| Okolitenko 1992 | 19.8 | 15 | та (43%) |
| Drozdovskyi 1991 | 18.0 | 15 | та (45%) |
| Stelmakh 2021 | 17.2 | 15 | та (54%) |
| Cherniatynskyi 1947 | 16.7 | 15 | та (64%) |
| Shevchuk 1991 | 16.6 | 15 | та (36%) |
| **LaPa (LLM)** | **14.0** | **10** | **та (80%)** |
| Nosenok 2020 | 13.1 | 15 | та (41%) |
| **GPT-5.2 (LLM)** | **7.1** | **15** | **ж (26%)** |
| **DeepL (LLM)** | **7.2** | **13** | **та (45%)** |

The three AI systems diverge internally on this feature. LaPa's total particle frequency (14.0/1k) is within human range, but 80% of its particles are a single word ("та") — far more concentrated than any human. GPT-5.2 and DeepL use **2.4× fewer** particles than the human average (7.1–7.2 vs. 16.9), the clearest fingerprint of machine translation. LaPa's Ukrainian fine-tuning partially closes this gap in frequency but not in diversity.

#### 3.5.3 Diminutive Morphology

Diminutive suffixes (-еньк-, -очк-, -ик, -оньк-, -ечк-) are a core expressive device in Ukrainian, signaling affection, irony, contempt, or intimacy.

| System | Diminutives /1k |
|--------|----------------|
| Shevchuk 1991 | 5.73 |
| Okolitenko 1992 | 5.14 |
| Drozdovskyi 1991 | 3.79 |
| Stelmakh 2021 | 3.04 |
| Cherniatynskyi 1947 | 3.01 |
| Nosenok 2020 | 2.88 |
| **GPT-5.2 (LLM)** | **2.78** |
| **LaPa (LLM)** | **2.61** |
| **DeepL (LLM)** | **2.28** |

All three AI systems cluster at the bottom. Humans average 3.9 diminutives per 1k words (excl. Dybko); AI averages 2.6 — a 1.5× gap. The LLM under-produces a fundamental Ukrainian morphological device.

#### 3.5.4 Function Word Poverty

Function words (conjunctions, prepositions, particles, pronouns) are content-independent markers of authorial style.

| System | Unique func words | Func word entropy (bits) |
|--------|------------------|------------------------|
| Humans (avg, excl. Dybko) | 182 | 5.93 |
| GPT-5.2 | 165 | 5.77 |
| DeepL | 158 | 5.66 |
| LaPa | 149 | 5.70 |

AI systems use 149–165 distinct function words vs. humans' average of 182. LaPa uses 18% fewer unique function words than the nearest human. Function word entropy is lowest for all AI systems — they rely on a narrower set of grammatical connectors. This is not about content vocabulary (which varies with translation strategy) but about the structural fabric of the prose.

### 3.6 Stylometric Distance

We computed pairwise Cosine Delta (Evert et al. 2017) using function-word lemma frequencies as features. Each translation is represented as a vector of function-word lemma relative frequencies (conjunctions, prepositions, particles, pronouns), z-score normalized across systems, then compared via cosine distance. We restrict features to function words because all texts translate the same source, making content words uninformative for style.

| System | Mean pairwise distance |
|--------|----------------------|
| Dybko 1984 | 1.162 |
| Cherniatynskyi 1947 | 1.146 |
| Drozdovskyi 1991 | 1.148 |
| LaPa (LLM) | 1.053 |
| Stelmakh 2021 | 1.110 |
| Okolitenko 1992 | 1.125 |
| Shevchuk 1991 | 1.110 |
| Nosenok 2020 | 1.094 |
| GPT-5.2 (LLM) | 1.050 |
| DeepL (LLM) | 1.044 |

`[cosine_delta_heatmap.png]`

The three AI systems cluster at the bottom (smallest mean pairwise distance = most stylistically central). Pairwise distances between AI systems (0.37–0.59) are far smaller than between any pair of human translators (0.97–1.28). By this content-independent measure, the three AI systems are stylistically near-identical.

**Interpretation:** AI systems' function-word *profiles* (which words they choose) cluster tightly together and near the corpus center. Their function-word *diversity* (how many they choose from) is the narrowest. They pick from the same pool as humans, but from a much smaller corner of it — and they all pick from the same corner.

### 3.7 Segment-Level Uniformity

| System | Word ratio std | Rank |
|--------|---------------|------|
| **LaPa (LLM)** | **0.128** | **1/10 (most uniform)** |
| Cherniatynskyi 1947 | 0.157 | |
| Shevchuk 1991 | 0.173 | |
| Stelmakh 2021 | 0.190 | |
| Nosenok 2020 | 0.216 | |
| **GPT-5.2 (LLM)** | **0.246** | |
| **DeepL (LLM)** | **0.264** | |
| Okolitenko 1992 | 0.273 | |
| Drozdovskyi 1991 | 0.335 | |
| Dybko 1984* | 0.614 | |

LaPa is the most uniform system. GPT-5.2 and DeepL are mid-pack on this measure — more uniform than some humans but not all. The violin plots (Section 3.1) tell a stronger story: all three AI systems' metric score distributions are tighter than any human's, confirming that AI produces uniformly "adequate" output without the peaks or valleys that characterize human stylistic choices.

### 3.8 Passage-Level Consistency

To test whether uniformity persists at longer spans, we split each translation into 27 passages (~50 segments each) and compute per-passage MATTR (lexical diversity, window=50) and em dash density.

`[passage_level_consistency.png]`

| System | MATTR mean | MATTR std | Em dash mean | Em dash std |
|--------|-----------|----------|-------------|------------|
| LaPa (LLM) | 0.904 | 0.010 | 3.7 | 3.8 |
| GPT-5.2 (LLM) | 0.909 | 0.013 | 11.6 | 7.2 |
| DeepL (LLM) | 0.902 | 0.011 | 9.7 | 5.6 |
| Humans (avg, excl. Dybko) | 0.914 | 0.012 | 16.8 | 9.2 |

AI systems are consistently impoverished (lower MATTR mean, lower em dash density) but their passage-to-passage variance is not uniquely extreme. H2c is confirmed at the segment level (violin distributions, word ratio std) but only partially at the passage level — making the human evaluation's dual-scale design (Section 5.1) essential.

---

## 4. Summary of Computational Claims

| Hypothesis | Status | Key Evidence |
|-----------|--------|-------------|
| H1a: Metrics reward fidelity over quality | **Supported** | All 3 AI systems rank top 3 on metrics, but bottom 3 on lexical diversity and expressiveness |
| H1b: Temporal bias | **Weak** | Cherniatynskyi (1947) scores mid-range, undermining a clean temporal story |
| H1c: Penalize cultural adaptation | **Needs human eval** | Cannot be proven computationally |
| H2a: AI lexically impoverished | **Confirmed** | All 3 AI systems below human range on MTLD, MATTR, hapax ratio, function-word types |
| H2b: AI lacks Ukrainian identity | **Confirmed** | LaPa dash-to-comma ratio 3.3× below nearest human; GPT/DeepL use 2.4× fewer discourse particles; all AI lowest on diminutives |
| H2c: AI artificially consistent | **Confirmed at segment level; partial at passage level** | AI violins are the thinnest; LaPa has lowest word-ratio std; passage-level variance is not uniquely non-human |
| H2d: AI convergence | **Strongly confirmed** | AI–AI pairwise LaBSE similarity 0.936 vs. H–H 0.684; Cosine Delta AI–AI distances 0.37–0.59 vs. H–H 0.97–1.28 |
| H3a: Training-data bubble | **Partially confirmed** | LaBSE amplifies AI's lead vs OpenAI, but direction holds on independent embeddings |
| H3b: Ref-free metrics can't distinguish strategies | **Supported** | COMETKiwi-22 and COMETKiwi-XL produce identical rankings (ρ = 0.964) |

---

## 5. Evaluation Experiment

The computational evidence identifies specific, testable patterns but cannot determine whether these patterns constitute measurement failure or reflect genuine quality differences. We split evaluation into two complementary tracks: human preference judgment (what only humans can do) and LLM-as-judge assessment (scalable fluency and accuracy rating).

### 5.1 Human Preference: TrueSkill Tournament

We adopt the pairwise preference tournament methodology from Romanyshyn et al. (2024), using the TrueSkill Bayesian rating system (Herbrich et al. 2006).

**Setup.** All 10 systems (7 human + 3 AI: GPT-5.2, DeepL, LaPa) are entered. We generate all 45 pairwise combinations per segment, draw a uniform random sample of segments, shuffle all pairs, and assign each annotator ~250–300 pairs. Each pair is judged once per annotator. Prior work shows ~180 judgments per system produce stable TrueSkill ratings with acceptable uncertainty (σ). The 3 AI–AI pairs serve as a convergence probe: if annotators cannot distinguish them (high tie rate), this independently confirms the embedding similarity finding (Section 3.2).

**Interface.** Each screen shows the English source sentence and two anonymized Ukrainian translations (left/right, randomized). The annotator clicks: **Left is better**, **Right is better**, or **Tie**. No Likert scales, no multi-dimensional rubrics — one binary-plus-tie decision per pair.

**Rating system.** TrueSkill initializes each system at μ = 25, σ = 25/3. Each judgment updates both systems' ratings: defeating a stronger opponent yields a larger rating gain; ties pull ratings toward each other. After all judgments, each system has a final μ (estimated quality) and σ (uncertainty). The output is a system-level leaderboard with confidence intervals — directly comparable to the metric ranking.

**Participants.** Native Ukrainian speakers (n ≥ 10). No translation expertise required — the task asks "which reads better as Ukrainian prose?" which general readers are qualified to judge. If professional translators are available, we run them as a separate group to test whether expertise shifts the ranking.

**Segment sampling.** Uniform random from the full 1,367-segment corpus. Stratification (by length, metric score, etc.) is unnecessary at this scale and risks cherry-picking accusations.

### 5.2 LLM-as-Judge: Fluency and Accuracy

To cover the dimensions that pairwise preference cannot isolate — accuracy against the source, and fluency as a monolingual quality — we use an LLM judge (GPT-4o or equivalent).

**Accuracy.** The LLM sees the English source and the Ukrainian translation and rates: *Does this translation accurately convey the meaning of the original?* (1–5). This runs exhaustively on all 1,367 segments × 10 systems (~13,670 judgments).

**Fluency.** The LLM sees only the Ukrainian translation (no source) and rates: *Does this read as natural, well-written Ukrainian?* (1–5). Same exhaustive coverage.

**Why LLM judge.** Accuracy and fluency are the dimensions MT metrics were trained to capture. Using an LLM judge (rather than human experts) to measure them creates a three-way comparison: metric scores vs. LLM-judge scores vs. human preference. If the LLM judge agrees with metrics but humans disagree, the divergence is specifically about what "quality" means — not about accuracy or fluency per se.

### 5.3 Analysis Plan

| Question | Data source | Method | Hypothesis |
|----------|------------|--------|------------|
| Does the human preference ranking match the metric ranking? | TrueSkill μ vs metric means | Spearman ρ, Kendall τ | H1a, H3b |
| Does the LLM-judge accuracy ranking match the metric ranking? | LLM accuracy means vs metric means | Spearman ρ | H1a |
| Does the LLM-judge fluency ranking diverge from human preference? | LLM fluency means vs TrueSkill μ | Spearman ρ | H3a |
| Do translations below AI fare better with humans than with metrics? | TrueSkill μ vs metric rank for bottom systems | Wilcoxon signed-rank | H1c |
| Do AI systems' TrueSkill σ differ from human translators? | TrueSkill σ per system | Descriptive | H2c |
| What is the AI–AI tie rate? | Tie frequency for AI–AI pairs vs. other pairs | χ² test | H2d |
| Is there a reader-vs-expert effect? (if two groups) | TrueSkill μ per group | Spearman ρ between group rankings | — |

**Statistical notes.** TrueSkill produces continuous μ and σ per system, enabling direct rank correlation with metric means. Wilcoxon signed-rank tests whether human ranks are systematically higher than metric ranks for the bottom half. The AI–AI tie rate tests H2d: if the three AI translations are perceptually indistinguishable, the tie rate should significantly exceed the baseline. The three-way comparison (metrics, LLM-judge, human preference) is the core result: agreement between any two against the third identifies whose "quality" definition is the outlier.

---

## 6. Expected Outcomes

1. **Human preference will diverge from metric rankings.** The TrueSkill leaderboard will not reproduce the metric ranking (Spearman ρ < 0.7). Specifically, AI systems will rank lower in human preference than their metric scores predict, and at least one lower-metric-scoring human translator will rank above them.

2. **LLM-judge accuracy will correlate with metrics; fluency will partially diverge.** Accuracy measures what metrics measure (source fidelity), so high correlation is expected. Fluency may diverge because metrics reward a modern, generic register while human readers prefer a more distinctive Ukrainian voice.

3. **AI systems will have the smallest TrueSkill σ.** Annotators will agree on AI quality more than on any human translator — not because AI is clearly good or clearly bad, but because it is consistently unremarkable. Human translators will show higher σ because their creative choices provoke genuine disagreement.

4. **AI–AI pairs will have elevated tie rates.** Since embedding similarity shows the three AI translations are near-identical (0.925–0.951), annotators should struggle to distinguish them, producing significantly more ties than for human–human or human–AI pairs.

5. **Translations scoring below AI on metrics will be partially rehabilitated.** Human preference will compress the gap between AI and the lower-ranked human translators, confirming that metrics exaggerate differences for translations that deviate from the normative band.

---

## 7. Limitations

1. **Single source text, single language pair.** All findings derive from English→Ukrainian translations of one novella.

2. **Small system count.** With 7 human translations and 3 AI systems, statistical power for system-level claims is inherently limited. TrueSkill's uncertainty quantification partially mitigates this.

3. **Three AI systems, one language direction.** While three architecturally distinct AI systems (general LLM, commercial NMT, domain-tuned LLM) strengthen generalizability over a single-LLM design, the finding may not hold for other language pairs or literary genres. A model with explicit literary training objectives (rather than general fine-tuning) might produce output with greater stylistic diversity.

4. **Embedding-space geometry is projection-dependent.** MDS centroid distances depend on the embedding model and dimensionality reduction method. We mitigate by cross-validating LaBSE with OpenAI embeddings.

5. **Sentence-level evaluation, not sustained reading.** Pairwise preference on individual sentences cannot capture the macro-experience of reading a chapter or a book — voice, rhythm, and narrative coherence. Our passage-level computational analysis (Section 3.8) partially addresses this but cannot replace a reading-level human judgment.

6. **LLM-as-judge has its own biases.** The LLM judge may share training-data biases with the MT metrics and the AI translators. If the LLM judge systematically favors AI output, this confound is informative (it strengthens H3a) but limits the judge's value as an independent accuracy/fluency signal.

7. **Human evaluation is proposed, not completed.** The computational evidence is suggestive but not conclusive. The core claims require human evaluation to confirm.

8. **Diminutive detection is approximate.** Regex-based diminutive counting may include false positives (words with diminutive-like suffixes that are not diminutives) and miss productive but irregular diminutive forms.

9. **XCOMET not yet computed for AI systems.** Reference-free XCOMET scores are available only for the 7 human translators. The two COMETKiwi metrics (22 and XL) both show the same AI > human ranking, so the gap is unlikely to change, but completeness requires running XCOMET on all 10 systems.

---

## 8. Implications

If confirmed, these findings have immediate practical consequences:

- **For MT evaluation research:** Current metrics should not be applied to literary translation without heavy disclaimers. A literary-specific evaluation metric is needed — one that rewards vocabulary richness, cultural sensitivity, and stylistic identity alongside semantic fidelity. The convergence of three architecturally distinct AI systems at the top of every metric ranking (while all three exhibit the same expressiveness deficits) suggests the problem is structural, not model-specific.

- **For AI-assisted translation:** AI systems may be useful as first-draft tools for literary translation, but their output requires substantial human post-editing to inject lexical diversity, cultural adaptation, and stylistic voice. The near-identity of GPT-5.2, DeepL, and LaPa output suggests that switching between AI systems does not solve the problem — the deficits are shared.

- **For translation studies:** The 74-year span of this corpus offers a natural experiment in how translation norms evolve. The metrics' preference for modern translations may reflect genuine quality improvement or temporal bias in the training data. The AI convergence finding adds a new dimension: if all AI systems converge on the same output, "AI translation" may represent a single point in the space of possible translations rather than a family of alternatives.

---

## Appendix A: Tools and Reproducibility

All computational analyses are fully reproducible from the scripts in this repository:

| Analysis | Script | Libraries |
|----------|--------|-----------|
| MT metrics (COMET, XCOMET, COMETKiwi) | `src/evaluator.py`, `src/main.py` | `unbabel-comet` |
| LaPa COMET scoring | `src/score_lapa.py` | `unbabel-comet` |
| Semantic similarity (LaBSE) | `src/semantic_similarity.py` | `sentence-transformers` |
| Semantic similarity (OpenAI) | `src/semantic_similarity_openai.py` | `openai` |
| Surface linguistics | `src/surface_linguistics.py` | `pymorphy3`, `lexicalrichness`, `sacrebleu`, `tokenize-uk` |
| Stylometry | `src/stylometry.py` | `pymorphy3`, `tokenize-uk`, `scipy` |
| Ukrainian expressiveness | `src/expressiveness.py` | `pymorphy3`, `tokenize-uk` |
| Passage-level consistency | `src/passage_level.py` | `lexicalrichness`, `tokenize-uk` |
| Human evaluation (TrueSkill) | TBD | `trueskill` |
| LLM-as-judge | TBD | `openai` |
| Visualization | `src/visualizer.py` | `matplotlib`, `seaborn` |

All results are stored in `results/` and plots in `plots/`.

## Appendix B: References

- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian Skill Rating System. *Advances in Neural Information Processing Systems 19*.
- Romanyshyn, M. et al. (2024). Setting up the Data Collection Pipeline for the Ukrainian Language. *Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP)*. ACL Anthology: 2024.unlp-1.9.
- Evert, S. et al. (2017). Understanding and explaining Delta measures for authorship attribution. *Digital Scholarship in the Humanities, 32*(suppl_2), ii4–ii16.
