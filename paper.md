# When Metrics Mistake Fidelity for Quality: Evaluating MT Metrics on Literary Translation

## Abstract

Automatic machine translation metrics — COMET, XCOMET, COMETKiwi, MetricX-24 — are the de facto standard for evaluating translation quality. But what do they actually measure? We investigate this question using a unique multilingual corpus: seven human Ukrainian translations of George Orwell's *Animal Farm* spanning 74 years (1947–2021), alongside three architecturally distinct AI systems (GPT-5.2, DeepL, and LaPa, a Ukrainian-tuned LLM). Across seven neural metrics, four reference-free and three reference-based, all three AI translations rank at the top. Yet stylometric analysis reveals that these same AI translations are lexically impoverished (−18% MTLD), underuse Ukrainian discourse particles (up to 2× fewer) and diminutive morphology (2.6× fewer), and converge on near-identical output (LaBSE pairwise similarity 0.941 vs. 0.711 for human pairs). A controlled LLM-as-a-judge experiment (750 pairs per condition) demonstrates a clean preference reversal: when the English source is visible, AI ranks first; when it is hidden and the judge evaluates literary quality alone, humans rise to the top and AI sinks. Human evaluation (1,034 pairwise judgments) confirms this pattern. We argue that current MT metrics reward semantic fidelity and surface fluency — properties AI systems optimize for — while failing to capture the lexical richness, cultural adaptation, and stylistic voice that define skilled literary translation.

---

## 1. Introduction

The past five years have seen a revolution in machine translation evaluation. Neural metrics trained on human quality judgments — COMET (Rei et al., 2020), COMETKiwi (Rei et al., 2022), XCOMET (Guerreiro et al., 2023), MetricX (Juraska et al., 2024) — consistently outperform surface-level metrics like BLEU (Papineni et al., 2002) and chrF (Popović, 2015) on meta-evaluation benchmarks. These metrics have become the standard for MT system development, shared tasks, and leaderboard rankings. When a system scores 0.85 on COMET, we trust it is producing high-quality translations.

But translation is not a monolithic task. Literary translation — with its emphasis on voice, register, cultural adaptation, and creative interpretation — occupies a fundamentally different space from the news and technical domains on which these metrics were trained. A literary translator does not optimize for source-closeness; they optimize for the reader's experience in the target language. The question we investigate is simple: **do MT metrics capture literary translation quality, or do they capture something else?**

We approach this question with a corpus that makes the comparison unusually sharp. George Orwell's *Animal Farm* has been translated into Ukrainian seven times between 1947 and 2021 — by translators working under Soviet censorship, in diaspora communities, and in modern independent Ukraine. These translations employ different strategies: literal fidelity, cultural adaptation, free rephrasing. To this set we add three AI translations from architecturally distinct systems: GPT-5.2 (a general-purpose LLM), DeepL (a commercial neural MT system), and LaPa (an LLM fine-tuned specifically for Ukrainian literary translation). If all three AI systems exhibit the same patterns despite different architectures and training regimes, the finding generalizes beyond any single model.

Our contributions:

1. **Comprehensive metric evaluation.** We compute seven neural MT metrics (four reference-free, three round-robin reference-based) across all ten systems, showing that AI occupies the top three positions on every metric.

2. **Multi-dimensional stylometric analysis.** We measure lexical diversity, discourse particle usage, diminutive morphology, surface overlap, cosine delta stylometric distance, and translation uniformity — revealing systematic AI deficits on every dimension of Ukrainian literary expressiveness.

3. **AI convergence.** Using cross-lingual embeddings (LaBSE) and round-robin neural metrics, we show that three architecturally distinct AI systems converge on near-identical output (pairwise similarity 0.941), while human translators genuinely diverge (0.711). AI translation is a single point in the space of possible translations, not a family of alternatives.

4. **Preference reversal.** Through a controlled LLM-as-a-judge experiment, we demonstrate that showing or hiding the English source completely reverses the ranking of AI vs. human translations — proving that what metrics reward (source fidelity) diverges from what constitutes literary quality.

---

## 2. Related Work

### 2.1 Neural MT Metrics

COMET (Rei et al., 2020) pioneered the use of pretrained cross-lingual encoders for MT evaluation, training regression models on Direct Assessment (DA) human judgments. Subsequent work expanded this approach: COMETKiwi (Rei et al., 2022) removed the need for reference translations; XCOMET (Guerreiro et al., 2023) scaled to larger models; MetricX (Juraska et al., 2024) applied instruction-tuned language models to the task. These metrics dominate the WMT Metrics Shared Task and have become standard in MT research. However, their training data is drawn almost exclusively from news and technical domains, and their meta-evaluation is conducted on the same domain distribution. Even for general-domain English–Ukrainian parallel data, Chaplynskyi and Zakharov (2025) found that an ensemble of six QE models explains only ~60% of the variance in human quality judgments, with a non-linear relationship between metric scores and human perception — suggesting that the gap widens further for literary text.

### 2.2 Literary Translation and MT

The evaluation of literary MT has received growing attention. Matusov (2019) noted that BLEU is inadequate for literary translation; Toral and Way (2018) found that neural MT produces more literal translations than human professionals. Kuzman et al. (2019) showed that MT output in literary domains was detectable by humans with high accuracy. More recently, several studies have examined LLM-based literary translation (Wang et al., 2023; Karpinska and Iyyer, 2023), finding that while fluency improves, creative and cultural dimensions remain weak. Our work complements these studies by providing fine-grained computational evidence for *why* metrics fail in the literary domain.

### 2.3 Stylometry and Translation

Computational stylometry — the quantitative analysis of writing style — has a long history in authorship attribution (Burrows, 2002; Evert et al., 2017). Cosine Delta, our primary stylometric measure, uses function-word frequencies as content-independent stylistic features. Applied to translations, stylometry can reveal whether different translators produce genuinely distinct voices or converge on a shared style. Rybicki (2012) used stylometric methods to show that translators preserve authorial signals; we extend this approach to compare human and AI translators' stylistic fingerprints.

### 2.4 Ukrainian Linguistic Features

Ukrainian literary prose employs several expressive devices with no direct English equivalents. Discourse particles (ж, таки, ось, бо, аж, ну, мов, наче) encode pragmatic nuances — emphasis, surprise, hedging, speaker attitude — that are central to natural Ukrainian writing (Shevelov, 1963). Diminutive morphology (suffixes -еньк-, -очк-, -ик, -оньк-, -ечк-) conveys affection, irony, contempt, or intimacy, functioning as a core expressive register (Wierzbicka, 1984). The systematic underuse of these features in translation would signal a failure of cultural adaptation.

---

## 3. Corpus

Our corpus comprises ten Ukrainian translations of George Orwell's *Animal Farm* (1945), aligned at the sentence level into 1,367 segments.

| ID | Year | Translator / System | Type | Ukrainian Title |
|----|------|---------------------|------|-----------------|
| T1 | 1947 | Ivan Cherniatynskyi | Human | Колгосп тварин |
| T2 | 1984 | Iryna Dybko | Human* | Хутір тварин |
| T3 | 1991 | Oleksii Drozdovskyi | Human | Скотоферма |
| T4 | 1991 | Yurii Shevchuk | Human | Ферма рай для тварин |
| T5 | 1992 | Natalia Okolitenko | Human | Скотохутір |
| T6 | 2020 | Bohdana Nosenok | Human | Колгосп тварин |
| T7 | 2021 | Viacheslav Stelmakh | Human | Колгосп тварин |
| T8 | — | LaPa (v0.1.2-instruct) | AI (tuned LLM) | — |
| T9 | — | GPT-5.2 | AI (general LLM) | — |
| T10 | — | DeepL | AI (commercial NMT) | — |

*Dybko's 1984 translation is a free cultural adaptation rather than a direct translation. We retain it as a sanity check (expected to rank last on fidelity metrics) but exclude it from group comparisons.

The seven human translations span 74 years and represent markedly different translation strategies. Cherniatynskyi (1947) translated in diaspora under conditions of limited editorial oversight; Dybko (1984) produced a free adaptation in Soviet Ukraine; Shevchuk (1991) is a professional literary translator known for careful register matching; Stelmakh (2021) represents contemporary professional translation practice.

The three AI systems represent distinct architectures: GPT-5.2 is a general-purpose LLM prompted for literary translation; DeepL is a commercial neural MT system; LaPa is an LLM fine-tuned on Ukrainian literary parallel data. If all three exhibit the same artifacts, the finding generalizes beyond any single model.

---

## 4. Methodology

### 4.1 Neural MT Metrics

We evaluate all ten translations using seven neural metrics spanning two paradigms:

**Reference-free metrics** require only the source sentence and the translation. They estimate quality without access to a human reference:
- **COMETKiwi-22** (`Unbabel/wmt22-cometkiwi-da`) — cross-lingual quality estimation
- **COMETKiwi-XL** (`Unbabel/wmt23-cometkiwi-da-xl`) — larger variant
- **XCOMET-XXL** (`Unbabel/XCOMET-XXL`) — multitask metric with error span detection
- **MetricX-24 QE** (`google/metricx-24-hybrid-xl-v2p6`) — instruction-tuned quality estimation

**Round-robin reference-based metrics** score each translation using every other translation as a pseudo-reference, producing a 10×10 pairwise score matrix. We report the mean score across all nine references:
- **COMET-22** (`Unbabel/wmt22-comet-da`)
- **XCOMET** (`Unbabel/XCOMET-XXL`)
- **MetricX-24** (`google/metricx-24-hybrid-xl-v2p6`)

For COMET-family metrics, higher scores indicate better quality. For MetricX, lower scores indicate better quality.

### 4.2 Semantic Similarity

We compute pairwise cosine similarity between all translations using LaBSE (Feng et al., 2022), a cross-lingual sentence embedding model. For each pair of systems, we embed all aligned segments with both systems, compute per-segment cosine similarity, and report the mean. We also compute each system's similarity to the English source.

### 4.3 Stylometric Analysis

We measure six dimensions of translational style:

- **Lexical diversity:** MTLD, MATTR, and hapax ratio, computed with `lexicalrichness` and `pymorphy3`. MTLD (Measure of Textual Lexical Diversity) is robust to text length; MATTR (Moving-Average Type-Token Ratio) uses a sliding window; hapax ratio measures the proportion of words used exactly once.
- **Discourse particles:** Frequency and diversity of Ukrainian discourse particles (ж, таки, ось, бо, аж, ну, мов, наче, etc.), detected by lemma matching with `pymorphy3`.
- **Diminutive morphology:** Frequency of diminutive suffixes (-еньк-, -очк-, -ик, -оньк-, -ечк-), detected by regex over morphologically analyzed tokens.
- **Surface overlap:** Pairwise chrF and BLEU between all translation pairs, computed with `sacrebleu`.
- **Cosine Delta:** Stylometric distance (Evert et al., 2017) using function-word lemma relative frequencies as features, z-score normalized, compared via cosine distance.
- **Word ratio uniformity:** Standard deviation of per-segment word count ratio (Ukrainian/English), measuring how consistently each system expands or compresses segments.

### 4.4 LLM-as-a-Judge Experiments

We conduct two controlled experiments using GPT-5.2 as the judge, with pairwise comparisons scored via TrueSkill (Herbrich et al., 2006):

**Experiment 1 — Translation quality.** The judge sees the English source sentence and two anonymized Ukrainian translations. The prompt asks: "You are an expert English-to-Ukrainian translator. Choose which translation is better." The judge responds with `system1`, `system2`, or `tie`.

**Experiment 2 — Literary quality.** The judge sees only two Ukrainian sentences (no English source). The prompt asks: "Choose the one that sounds more literary — as if written by a skilled Ukrainian author for a published book. Judge only literary quality: naturalness, expressiveness, and stylistic richness of the Ukrainian language." The judge responds with `system1`, `system2`, or `tie`.

The only difference between experiments is the presence/absence of the English source and the framing of the quality criterion. By comparing the resulting TrueSkill rankings, we isolate the effect of source fidelity on quality judgment.

### 4.5 Sampling Design and Human Evaluation

All three evaluations — both LLM judge experiments and the human evaluation — draw from the same pool: 1,128 valid segments × 45 system pairs = 50,760 items, globally shuffled with a fixed seed. This ensures that even early subsets cover all 45 system pairs approximately uniformly. Left/right presentation order is randomized per pair to control for position bias. TrueSkill ratings (Herbrich et al., 2006) are computed with default parameters (μ₀ = 25, σ₀ = 25/3).

For the LLM judge, ~750 pairs were evaluated per experiment (~17 judgments per system pair). Rankings stabilized between the 500- and 750-pair checkpoints (max win-rate delta < 1 percentage point).

For the human evaluation, we adopt the pairwise preference tournament of Romanyshyn et al. (2024). Annotators see the English source and two anonymized Ukrainian translations, then select *Left is better*, *Right is better*, or *Tie*. Each pair is judged by one annotator. At 1,034 matches, each system participates in ~207 comparisons (σ ≈ 0.75–0.77 for most systems), with the top (Stelmakh) and bottom (Dybko) clearly separated and a compressed mid-pack (§5.4).

---

## 5. Results

We present results in four stages. First, we establish that every automatic metric ranks AI at the top (§5.1). Second, we show that these AI translations are near-identical to each other and unusually close to the English source (§5.2). Third, we demonstrate that the same AI translations are systematically impoverished on every dimension of Ukrainian literary expressiveness (§5.3). Finally, we show that removing the English source from the evaluation reverses the ranking entirely (§5.4).

### 5.1 Every Metric Says AI Wins

We computed four reference-free and three round-robin reference-based neural metrics across all ten systems. The result is unambiguous: **all three AI systems occupy the top three positions on every metric**. The best human translator (Stelmakh, 2021) consistently ranks fourth.

On COMETKiwi-22, AI systems average 0.812 vs. 0.724 for humans (excluding Dybko) — a gap of +0.088. On MetricX-24 QE (where lower is better), AI averages 3.48 vs. humans' 5.45. The gap holds across COMETKiwi-XL, XCOMET-XXL, and all three round-robin metrics (COMET-22, XCOMET, MetricX-24). No metric ranks any human translator above any AI system.

[ref_free_comparison.png]

[round_robin_comparison.png]

LaPa — fine-tuned specifically for Ukrainian literary translation — tops the COMETKiwi rankings, but the margin over GPT-5.2 (a general-purpose LLM) is 0.008. Domain tuning yields negligible improvement on metric scores.

Dybko (1984), our sanity check, ranks dead last on every metric. Her free cultural adaptation departs so far from the source that metrics penalize it maximally. This confirms the metrics are sensitive; the question is what they are sensitive *to*.

### 5.2 AI Systems Converge on the Same Translation

The round-robin heatmaps reveal a striking visual pattern. The 3×3 AI-AI block (bottom-right) is uniformly hot — the highest pairwise scores in the entire matrix — while the 7×7 human-human block shows much more variation. Dybko's row is cold throughout.

[heatmap_xcomet.png]

[heatmap_metricx_24.png]

We quantify this convergence across five independent measures:

| Measure | AI–AI | Human–Human | Gap |
|---------|:-----:|:-----------:|:---:|
| LaBSE cosine similarity | **0.941** | 0.711 | +0.230 |
| XCOMET round-robin | **0.886** | 0.627 | +0.259 |
| COMET-22 round-robin | **0.881** | 0.750 | +0.131 |
| MetricX-24 round-robin | **3.38** | 7.61 | −4.23 |
| chrF (surface overlap) | **43.1** | 33.7 | +9.4 |

*Table 1. AI–AI vs. Human–Human pairwise scores. For MetricX-24, lower = more similar. Bold = more similar group.*

The individual AI–AI LaBSE pairs — LaPa–DeepL (0.952), GPT-5.2–DeepL (0.940), LaPa–GPT-5.2 (0.932) — represent near-identical translations. The convergence holds on every measure, from neural embeddings to raw character n-grams. Three architecturally distinct systems have arrived at the same output.

[convergence_round_robin.png]

AI systems are also measurably closer to the English source than any human translator. On LaBSE, all three AI systems exceed 0.845 similarity to the source; the closest human (Cherniatynskyi, 1947) reaches 0.783. The human average (excluding Dybko) is 0.733. AI translations are structurally more "English" than any human translation.

[source_similarity.png]

[pairwise_heatmap.png]

### 5.3 What the Metrics Miss: Stylometric Analysis

The same AI translations that dominate every metric are systematically impoverished on every dimension of Ukrainian literary expressiveness.

**Lexical diversity.** AI systems average an MTLD of 311 vs. 377 for humans (excluding Dybko) — an 18% gap. DeepL (301) and LaPa (303) fall below every human translator except Dybko. The hapax ratio (words used exactly once) tells the same story: AI 0.182 vs. human 0.215, a 15% deficit. AI translations cycle through a narrower vocabulary, concentrating 39.4% of their text in the 100 most common words (vs. 37.7% for humans).

[lexical_diversity.png]

**Discourse particles.** Ukrainian discourse particles (ж, таки, ось, бо, аж, ну, мов, наче) encode pragmatic nuances — emphasis, surprise, hedging — with no direct English equivalents. They must be *added* by the translator. GPT-5.2 and DeepL produce approximately 2× fewer particles than the human average. LaPa partially closes the frequency gap thanks to Ukrainian fine-tuning, but concentrates overwhelmingly on a single particle (та) — a sign of pattern memorization, not pragmatic competence.

[discourse_particles.png]

**Diminutive morphology.** Diminutive suffixes (-еньк-, -очк-, -ик, -оньк-) are a core expressive device in Ukrainian, signaling affection, irony, or intimacy. AI systems average 0.47 diminutives per thousand tokens; humans average 1.23 — a **2.6× gap**. All three AI systems rank at the bottom, below every human translator. LaPa, despite literary fine-tuning, is indistinguishable from GPT-5.2 and DeepL on this measure.

[diminutives.png]

**Cosine Delta (stylometric fingerprint).** Using function-word lemma frequencies — content-independent markers of translatorial voice — Cosine Delta (Evert et al., 2017) measures how stylistically distinct each system is from the others. AI systems have the smallest mean pairwise distance (DeepL 1.058, GPT-5.2 1.059, LaPa 1.068), placing them closest to each other and to the corpus centroid. Human translators range from 1.095 (Nosenok) to 1.160 (Dybko). By this measure, AI systems occupy the same narrow corner of the stylistic space.

[cosine_delta.png]

**Segment-level uniformity.** LaPa (σ = 0.127) and DeepL (σ = 0.132) are the two most uniform systems, maintaining near-constant word count ratios across segments. Human translators vary more — they expand descriptive passages and compress dialogue, adapting to content. AI produces uniformly adequate output without the peaks and valleys that characterize human stylistic choices.

[word_ratio.png]

### 5.4 The Preference Reversal

This is the central experiment. We ran two LLM-as-a-judge experiments (~750 pairwise comparisons each, with rankings stable between the 500- and 750-pair checkpoints) with identical setup except for one variable: the presence of the English source.

- **Experiment 1 (Translation):** Judge sees the English source + two Ukrainian translations. Prompt: *"Choose the better translation."*
- **Experiment 2 (Literary):** Judge sees only two Ukrainian sentences. Prompt: *"Choose the one that sounds more literary — as if written by a skilled Ukrainian author."*

We compute TrueSkill ratings from each experiment and compare them to metric rankings and human evaluation (1,034 matches). The results are in Table 2; the ranking shifts for AI systems are in Table 3.

| System | Metrics | LLM: Translation | LLM: Literary | Human Eval |
|--------|:-------:|:-----------------:|:-------------:|:----------:|
| **GPT-5.2** | **#2** | **#1** (30.3) | **#6** (24.4) | **#5** (25.3) |
| **DeepL** | **#3** | **#3** (29.2) | **#8** (23.8) | **#2** (26.4) |
| **LaPa** | **#1** | **#4** (27.4) | **#9** (22.4) | **#7** (24.9) |
| Stelmakh 2021 | #4 | #2 (29.8) | #2 (28.6) | #1 (27.1) |
| Shevchuk 1991 | #5 | #5 (26.7) | #3 (26.6) | #3 (26.3) |
| Drozdovskyi 1991 | #8 | #7 (24.0) | #1 (29.5) | #6 (25.2) |
| Okolitenko 1992 | #9 | #9 (21.4) | #4 (25.2) | #9 (23.3) |
| Nosenok 2020 | #7 | #8 (23.5) | #5 (24.7) | #4 (25.3) |
| Cherniatynskyi 1947 | #6 | #6 (25.0) | #7 (24.3) | #8 (24.1) |
| Dybko 1984 | #10 | #10 (13.9) | #10 (21.4) | #10 (18.1) |

*Table 2. System rankings across four evaluation paradigms. TrueSkill μ in parentheses. Bold = AI systems. Human eval: 1,034 matches. LLM judges: ~750 pairs each.*

| AI system | Metric rank | Translation judge | Literary judge | Shift |
|-----------|:-----------:|:-----------------:|:--------------:|:-----:|
| GPT-5.2 | #2 | #1 | #6 | ↓5 |
| DeepL | #3 | #3 | #8 | ↓5 |
| LaPa | #1 | #4 | #9 | ↓5 |

*Table 3. AI rank shifts when the English source is removed from evaluation.*

The reversal is uniform: every AI system drops exactly five positions when the source is hidden. The top five positions in the literary ranking are all human translators.

[trueskill_comparison.png]

**The Drozdovskyi effect.** The most dramatic individual reversal belongs to Drozdovskyi (1991). He ranks #8 on COMETKiwi-22 (0.693) and #7 on the translation judge — firmly in the bottom half. But on the literary judge, he leaps to #1 (μ = 29.5), surpassing every system including Stelmakh. His translation is rich in discourse particles (10.1/1k — the highest in the corpus) and diminutives (1.36/1k). These are exactly the features metrics penalize and literary judgment rewards. Drozdovskyi is the preference reversal embodied in a single translator.

**Stelmakh as bridge.** Stelmakh (2021) is the only system that ranks in the top two across every non-metric ranking: #1 in human eval (27.1), #2 in literary judge (28.6), #2 in translation judge (29.8). On metrics, he ranks #4 — the best human. He represents the rare translator whose work satisfies both fidelity-oriented and literary-oriented evaluation: semantically faithful enough for metrics, stylistically rich enough for readers.

**Human evaluation aligns with neither judge cleanly.** The 1,034-match human evaluation places Stelmakh clearly first (μ = 27.1), but positions #2–#7 form a compressed cluster (μ = 24.9–26.4) with overlapping confidence intervals. DeepL ranks #2 and Shevchuk #3 — both above GPT-5.2 (#5) and LaPa (#7). Humans value fidelity enough to keep DeepL near the top, but not enough to replicate the metric ranking where all three AI systems dominate.

**Dybko anchors the bottom.** She ranks last in every paradigm without exception — metrics, both LLM judges, and human eval. That all instruments agree she is last confirms they are functioning. The question is not whether they detect bad translations, but whether they can distinguish among the good ones.

---

## 6. Discussion

### 6.1 What Do Metrics Actually Measure?

Our results suggest a clear answer: neural MT metrics measure **semantic fidelity to the source** and **surface fluency in the target language**. These are the properties AI systems optimize for (via RLHF, parallel training data, and instruction tuning), and these are the properties on which AI systems score highest.

This is not a flaw in the metrics per se — semantic fidelity and fluency are genuine dimensions of translation quality. The problem is one of **construct validity**: the metrics claim to measure "translation quality" but actually measure a subset of quality that is systematically correlated with AI output. In the literary domain, this subset is insufficient. A translation can be semantically faithful and fluent while being lexically impoverished, culturally thin, and stylistically anonymous.

The preference reversal (Section 5.5) is the strongest evidence for this claim. When the source is visible, the judge has access to the same information the metrics use, and agrees with them. When the source is hidden, the judge must evaluate the Ukrainian text on its own merits — and the ranking inverts. The quality that survives without source access is a different quality from what metrics capture.

### 6.2 The Convergence Problem

Perhaps the most striking finding is the convergence of three architecturally distinct AI systems. GPT-5.2, DeepL, and LaPa — a general-purpose LLM, a commercial NMT system, and a domain-tuned LLM — produce translations with pairwise LaBSE similarity of 0.941. By every measure (neural metrics, surface overlap, cross-lingual embeddings, cosine delta), they are closer to each other than any pair of human translators is to each other.

This has implications beyond evaluation. If AI translation converges on a single point in the space of possible translations, then:

1. **Switching between AI systems does not increase diversity.** An editor hoping that DeepL and GPT produce usefully different alternatives will find they do not.

2. **Domain tuning has limited impact on style.** LaPa, fine-tuned specifically for Ukrainian literary translation, produces output nearly indistinguishable from GPT-5.2 on stylometric measures. Fine-tuning shifts the metric scores marginally but does not produce a distinct voice.

3. **AI translation is not "a family of translations" but "one translation with three names."** The diversity that characterizes human literary translation — where seven translators produce seven genuinely different texts — is absent from AI output.

### 6.3 The Ukrainian Expressiveness Gap

The stylometric deficits we document are not arbitrary. Discourse particles, diminutive morphology, and lexical diversity are the specific features that Ukrainian literary scholars and readers identify as markers of skilled prose. Their absence in AI translations is not a matter of personal preference but of cultural competence.

The diminutive gap (2.6×) is particularly informative. Diminutives in Ukrainian are context-dependent pragmatic devices — the same base word can take different diminutive suffixes to convey different attitudes. This requires understanding not just the words but the emotional register of the scene. AI systems' failure to produce diminutives at human rates suggests a shallow engagement with the affective dimension of the text.

Similarly, the discourse particle deficit reveals that AI systems translate *what is said* but not *how it is said*. English has no direct equivalents for ж, таки, or ось, so these particles must be *added* by the translator based on pragmatic understanding. Human translators add them naturally; AI systems do not.

### 6.4 Implications

**For MT evaluation research.** Current metrics should not be applied to literary translation without explicit caveats. A literary-specific evaluation metric is needed — one that rewards vocabulary richness, cultural sensitivity, and stylistic identity alongside semantic fidelity.

**For AI-assisted translation.** AI output may be useful as a first draft, but requires substantial post-editing to inject lexical diversity, cultural markers, and stylistic voice. The convergence finding means that switching between AI providers does not solve this problem.

**For translation studies.** The 74-year span of our corpus provides a natural experiment in how translation norms evolve. The metrics' preference for modern, source-close translations may reflect genuine quality improvements *or* temporal bias in their training data. The fact that Cherniatynskyi (1947) scores mid-range on all metrics — neither top nor bottom — weakly undermines a pure temporal bias explanation.

---

## 7. Limitations

1. **Single source text, single language pair.** All findings derive from English → Ukrainian translations of one novella. Generalization to other language pairs, genres, or longer texts requires further work.

2. **Small system count.** With seven human translations and three AI systems, statistical power for system-level claims is limited. TrueSkill's uncertainty quantification partially mitigates this.

3. **Human evaluation mid-pack resolution.** The 1,034-match human eval clearly separates top (Stelmakh) and bottom (Dybko), but positions #2–#7 form a compressed cluster with overlapping confidence intervals (σ ≈ 0.76, spread ≈ 1.5 μ points).

4. **LLM-as-a-judge shares training biases.** GPT-5.2 evaluating GPT-5.2's own output introduces a potential self-preference bias. That the literary judge still demotes AI translations makes this concern less acute, but an independent human-only evaluation is needed for confirmation.

5. **Diminutive detection is approximate.** Regex-based diminutive counting may include false positives (words with diminutive-like suffixes) and miss irregular forms.

6. **Sentence-level evaluation.** Pairwise preference on individual sentences cannot capture the macro-experience of reading sustained prose — voice, rhythm, and narrative coherence over chapters.

---

## 8. Conclusion

We set out to ask whether MT metrics capture literary translation quality. The answer is no — they capture something related but importantly different. Across seven neural metrics, three AI translations consistently rank at the top. Yet these same translations are lexically impoverished (−18% MTLD), culturally thin (2.6× fewer diminutives, ~2× fewer discourse particles), and converge on near-identical output (LaBSE 0.941). When a judge evaluates the translations without access to the source — judging only whether the text sounds like skilled Ukrainian literary prose — the AI advantage disappears and human translators take the top five positions. The reversal is sharpest for Drozdovskyi (1991), who ranks #8 on metrics but #1 on the literary judge.

The problem is not that the metrics are broken. They accurately measure what they were trained to measure: semantic fidelity to the source and surface fluency in the target. But literary translation requires more than fidelity and fluency. It requires voice, cultural adaptation, and expressive richness — qualities that current metrics cannot detect and that AI systems do not produce.

The convergence of three architecturally distinct AI systems on the same output strengthens this conclusion. The pattern is not a quirk of one model but a structural property of current AI translation: these systems all optimize for the same objective and arrive at the same solution — a single, lexically impoverished, culturally thin, metrically excellent point in the vast space of possible translations.

---

## References

- Burrows, J. (2002). 'Delta': A measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing, 17*(3), 267–287.
- Chaplynskyi, D., & Zakharov, K. (2025). A framework for large-scale parallel corpus evaluation: Ensemble quality estimation models versus human assessment. *Proceedings of the Fourth Ukrainian Natural Language Processing Workshop (UNLP 2025)*, 73–85. ACL.
- Evert, S., Proisl, T., Jannidis, F., Reger, I., Pielström, S., Schöch, C., & Vitt, T. (2017). Understanding and explaining Delta measures for authorship attribution. *Digital Scholarship in the Humanities, 32*(suppl_2), ii4–ii16.
- Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. *Proceedings of ACL 2022*.
- Guerreiro, N. M., Rei, R., Stanton, D., Farinhas, A., Fernandes, P., Martins, A. F. T., & Blunsom, P. (2023). xCOMET: Transparent machine translation evaluation through fine-grained error detection. *arXiv preprint arXiv:2310.10482*.
- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian skill rating system. *Advances in Neural Information Processing Systems 19*.
- Juraska, J., Finkelstein, M., Deutsch, D., Siddhant, A., Miber, M., & Freitag, M. (2024). MetricX-24: The Google submission to the WMT 2024 Metrics Shared Task. *Proceedings of WMT 2024*.
- Karpinska, M., & Iyyer, M. (2023). Large language models effectively leverage document-level context for literary translation, but critical errors persist. *Proceedings of WMT 2023*.
- Kuzman, T., Vintar, Š., & Arčan, M. (2019). Neural machine translation of literary texts from English to Slovene. *Proceedings of the Qualities of Literary Machine Translation*, 1–9.
- Matusov, E. (2019). The challenges of using neural machine translation for literature. *Proceedings of the Qualities of Literary Machine Translation*, 10–19.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: A method for automatic evaluation of machine translation. *Proceedings of ACL 2002*, 311–318.
- Popović, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. *Proceedings of the Tenth Workshop on Statistical Machine Translation*, 392–395.
- Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A neural framework for MT evaluation. *Proceedings of EMNLP 2020*.
- Rei, R., Treviso, M., Guerreiro, N. M., Zerva, C., Farinha, A. C., Marber, C., de Souza, J. G. C., Glushkova, T., Alber, D., Coheur, L., Lavie, A., & Martins, A. F. T. (2022). CometKiwi: IST-Unbabel 2022 submission for the quality estimation shared task. *Proceedings of WMT 2022*.
- Romanyshyn, M., Chaplynskyi, D., & Lukashchuk, N. (2024). Setting up the data collection pipeline for the Ukrainian language. *Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP)*, ACL 2024.
- Rybicki, J. (2012). The great mystery of the (almost) invisible translator: Stylometry in translation. *Quantitative Methods in Corpus-Based Translation Studies*, 231–248.
- Shevelov, G. Y. (1963). *The Syntax of Modern Literary Ukrainian: The Simple Sentence.* Mouton.
- Toral, A., & Way, A. (2018). What level of quality can neural machine translation attain on literary text? *Translation Quality Assessment*, 263–287.
- Wang, L., Lyu, C., Ji, T., Zhang, Z., Yu, D., Shi, S., & Tu, Z. (2023). Document-level machine translation with large language models. *Proceedings of EMNLP 2023*.
- Wierzbicka, A. (1984). Diminutives and depreciatives: Semantic representation for derivational categories. *Quaderni di Semantica, 5*(1), 123–130.

---

## Appendix A: Reproducibility

All computational analyses are fully reproducible from the scripts in this repository:

| Analysis | Script | Key Libraries |
|----------|--------|---------------|
| COMET / XCOMET metrics | `src/comet_evaluate.py` | `unbabel-comet` |
| MetricX-24 metrics | `src/metricx_evaluate.py` | `transformers`, `mt5` |
| Semantic similarity (LaBSE) | `src/semantic_similarity.py` | `sentence-transformers` |
| Stylometry (all metrics) | `src/stylometry/` | `pymorphy3`, `lexicalrichness`, `sacrebleu`, `tokenize-uk` |
| LLM-as-a-judge (translation) | `src/llm_judge.py` | `openai` |
| LLM-as-a-judge (literary) | `src/llm_judge_literary.py` | `openai` |
| TrueSkill rankings | `src/trueskill_rank.py` | `trueskill` |
| Neural metrics visualization | `src/neural_metrics_analysis.py` | `matplotlib` |

Results are stored in `results/` and plots in `plots/`. All code is available at [repository URL].

## Appendix B: Full Plot Index

| Figure | Description | Path |
|--------|-------------|------|
| Fig. 1 | Reference-free metric comparison | `plots/neural_metrics/ref_free_comparison.png` |
| Fig. 2 | Round-robin metric comparison | `plots/neural_metrics/round_robin_comparison.png` |
| Fig. 3 | COMET-22 pairwise heatmap | `plots/neural_metrics/heatmap_comet_22.png` |
| Fig. 4 | XCOMET pairwise heatmap | `plots/neural_metrics/heatmap_xcomet.png` |
| Fig. 5 | MetricX-24 pairwise heatmap | `plots/neural_metrics/heatmap_metricx_24.png` |
| Fig. 6 | Round-robin convergence | `plots/neural_metrics/convergence_round_robin.png` |
| Fig. 7 | LaBSE pairwise similarity heatmap | `plots/labse/pairwise_heatmap.png` |
| Fig. 8 | LaBSE cluster averages (AI-AI, H-AI, H-H) | `plots/labse/cluster_averages.png` |
| Fig. 9 | LaBSE similarity to English source | `plots/labse/source_similarity.png` |
| Fig. 10 | Lexical diversity (MTLD, Hapax, Top-100) | `plots/stylometry/lexical_diversity.png` |
| Fig. 11 | Discourse particles | `plots/stylometry/discourse_particles.png` |
| Fig. 12 | Diminutive morphology | `plots/stylometry/diminutives.png` |
| Fig. 13 | chrF / BLEU surface overlap | `plots/stylometry/chrf_bleu.png` |
| Fig. 14 | Cosine Delta heatmap | `plots/stylometry/cosine_delta.png` |
| Fig. 15 | Word ratio uniformity | `plots/stylometry/word_ratio.png` |
| Fig. 16 | Stylometry convergence summary | `plots/stylometry/convergence_summary.png` |
| Fig. 17 | TrueSkill comparison (3 sources) | `plots/trueskill/trueskill_comparison.png` |
