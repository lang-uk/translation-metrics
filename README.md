## Semantic Fidelity Versus Literary Quality: A Construct Validity Study of Neural Machine Translation Metrics

Evaluating MT metrics adequacy for literary translation: 7 human Ukrainian translations of *Animal Farm* (1947–2021) + 3 LLM translations.

## Reproducing Results

```bash
pip install pymorphy3 lexicalrichness sacrebleu tokenize-uk sentence-transformers matplotlib seaborn scipy
```

### Analysis scripts

Run any script from the repo root:

```bash
python src/surface_linguistics.py
python src/stylometry.py
python src/expressiveness.py
python src/passage_level.py
```

### MT metric evaluation

Requires `unbabel-comet` and HuggingFace authentication for gated models.

```bash
python src/main.py data/17099743/Animal_farm.tmx --mode all --custom-translations data/lapa_translations_combine.json --newlines-strategy combine --visualize
```

## Data

- `data/17099743/Animal_farm.tmx` — aligned TMX: English original + 7 Ukrainian translations
- `data/lapa_translations_combine.json` — lapa LLM translation
- `results/unite-models/parsed_translations.json` — all 8 translations, segment-aligned (used by analysis scripts)

## Plots

All plots are saved to `plots/`.
