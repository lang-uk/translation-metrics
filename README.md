# translation-metrics

`translation-metrics` — це інструментарій для роботи з багатомовними корпусами текстів і перекладів. Він дозволяє аналізувати якість перекладів, порівнювати різні моделі перекладу, а також візуалізувати результати аналізу (матриці порівняння чи violin plots).

## Використання

### Основі модулі

`tmx_parser.py` – модуль для парсингу TMX файлу, дозволяє експортувати переклади в формати JSON, CSV, TXT.

`analyze_tmx_segments.py` – скрипт для аналізу кількості сегментів в TMX файлі і порівняння з текстовим файлом (використовувався для виявлення невідповідностей у кількості сегментів через newlines в одному реченні з оригінального TMX).

`run_metricx_predict.sh` – скрипт для запуску передбачень моделями MetricX.

`run_metricx_evaluate.sh` – скрипт для запуску оцінювання моделями MetricX.

`metricx_converter.py` – скрипт для конвертації тексту в формат для оцінювання моделями MetricX (приймає на вхід TMX файл із перекладами і створює файли у форматі для оцінювання моделями MetricX).

`custom_translations_loader.py` – модуль для завантаження кастомних перекладів (не з оригінального TMX файлу) з різних форматів (JSON, JSONL, CSV, Parquet, TXT).

`metricx` – репозиторій `https://github.com/google-research/metricx`, підготований до evaluation-запусків.

Приклад використання модуля `main.py`:

- оцінювання всіма метриками для усіх перекладів з візуалізацією:

```bash
python main.py 17099743/Animal_farm.tmx --mode all --custom-translations lapa_translations_combine.json --newlines-strategy combine --visualize
```

`prepare_xcomet.sh` – скрипт для підготовки середовища для роботи з xCOMET.

`visualizer.py` – основний модуль для візуалізації результатів оцінювання.

`evaluator.py` – основний модуль для оцінювання перекладів.


### Дані

`lapa_translations_fixed.txt` – файл з перекладами від `lapa-llm/lapa-v0.1.2-instruct` (одного разу при інпуті **---** модель згенерувала відповідь з великою кількістю newlines, тому їх прибрали, щоб все було в одному рядку).

`lapa_model_metricx_23_qe.jsonl` – переклади від `lapa-llm/lapa-v0.1.2-instruct`, підготовані для оцінювання моделями MetricX-23-QE.

`lapa_model_metricx_24_qe.jsonl` – переклади від `lapa-llm/lapa-v0.1.2-instruct`, підготовані для оцінювання моделями MetricX-24-QE.

`17099743/Animal_farm.tmx` – оригінальний TMX файл з текстом Animal Farm (EN оригінал + UK переклади від різних авторів).

`17099743/Animal_farm.json` – JSON файл з текстом Animal Farm (один елемент відповідає одному реченню з TMX-оригіналу).

`17099743/Animal_farm_EN_1945_George_Orwell_Animal_Farm.txt` – аналог JSON-файлу, але у форматі `.txt` (один рядок – одне речення з оригіналу).

`17099743/Animal-Farm_preface.tmx` – TMX файл з перекладами вступу до Animal Farm.

`metricx_data/` – папка з файлами у форматі для попарної (round-robin) оцінки моделями MetricX (переклад_1 vs переклад_2).

### Результати оцінювання

`results/comet` – повні результати оцінювання метриками:
  - `Unbabel/wmt23-cometkiwi-da-xl`
  - `Unbabel/XCOMET-XL`
  - `Unbabel/wmt22-cometkiwi-da`
  - `Unbabel/wmt22-comet-da`

`results/metricx` – папка з результатами оцінювання моделями MetricX (оцінювання попарне, round-robin).

`results/metricx-qe` – папка з результатами оцінювання моделями MetricX-QE (оцінювання vs original text, quality estimation only).

`results/unite-models` – аналогічні результати для моделей `Unbabel/unite-mup` та `Unbabel/wmt22-unite-da`.