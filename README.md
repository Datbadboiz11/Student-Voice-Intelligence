# Student Voice Intelligence

He thong NLP phan tich phan hoi sinh vien tieng Viet. Project hien tai da hoan thanh pipeline data, EDA, label enrichment, baseline classification va fine-tune Transformer cho sentiment/topic.

## Muc tieu

Project xu ly cac phan hoi dang text va du doan:

- `sentiment`: positive / neutral / negative
- `topic`: nhom chu de phan hoi
- `toxic`: co ngon ngu doc hai/xuc pham hay khong
- `urgency`: muc do can xu ly low / medium / high

## Trang thai hien tai

Da hoan thanh:

- Data merge NEU-ESC + UIT-VSFC
- EDA va bao cao du lieu
- Rule-based enrichment cho toxic, urgency, topic_group
- LLM review cho urgency candidates
- Baseline models bang TF-IDF + Logistic Regression / Linear SVM
- Notebook demo inference tren vai cau mau
- Fine-tune Transformer cho `sentiment_std_3class`
- So sanh 4 Transformer sentiment: XLM-R, PhoBERT-base-v2, ViDeBERTa, PhoBERT-large
- Chot `vinai/phobert-base-v2` lam model sentiment chinh
- Fine-tune Transformer cho `topic_group`
- So sanh 3 bien the PhoBERT topic: full class weight, no weight, sqrt class weight
- Chot `topic_phobertv2_noweight` lam model topic chinh

Dang lam tiep:

- Phase 5+: toxic/urgency Transformer, multi-task learning, semantic search, RAG, dashboard

## Cau truc thu muc

```text
Student Voice Intelligence/
|
|-- notebook/
|   |-- data/
|   |   |-- data_merge.ipynb
|   |   |-- eda.ipynb
|   |   |-- label_enrichment.ipynb
|   |   `-- llm_review_urgency.ipynb
|   |
|   `-- baseline/
|       |-- baseline_models.ipynb
|       `-- 06_baseline_inference_demo.ipynb
|   |
|   `-- transformer/
|       |-- train_xlmr_sentiment.ipynb
|       |-- train_phobertv2_sentiment.ipynb
|       |-- train_videberta_sentiment.ipynb
|       |-- train_phobertlarge_sentiment.ipynb
|       |-- train_phobertv2_topic.ipynb
|       |-- train_phobertv2_topic_noweight.ipynb
|       `-- train_phobertv2_topic_sqrt_weight.ipynb
|
|-- data/
|   `-- processed/              # ignored by git
|
|-- datasets/                   # ignored by git
|-- outputs/
|   |-- reports/                # report ket qua
|   |-- figures/                # ignored by git
|   `-- models/                 # ignored by git
|
|-- PLAN.md
|-- note.txt
|-- .env.example
`-- .gitignore
```

## Data

Project dung 2 dataset:

- NEU-ESC: `hung20gg/NEU-ESC`
- UIT-VSFC: `uitnlp/vietnamese_students_feedback`

Do data CSV va model artifacts co the lon, repo dang ignore:

- `datasets/**/*.csv`
- `data/processed/*.csv`
- `outputs/models/`
- `outputs/figures/`

Nguoi dung moi can tai/chuan bi data goc truoc khi chay pipeline.

## File data chinh

Sau khi chay day du pipeline, file data chinh la:

```text
data/processed/student_voice_enriched_reviewed.csv
```

File nay gom:

- data da merge va chuan hoa
- `sentiment_std_3class`
- `topic_group`
- `is_toxic`
- `urgency_level_final`

Khi train PhoBERT, notebook se tao hoac dung lai file cache:

```text
data/processed/student_voice_enriched_reviewed_phobert.csv
```

File cache nay co them cot:

```text
text_phobert
```

## Thu tu chay notebook

Chay theo thu tu:

```text
notebook/data/data_merge.ipynb
notebook/data/eda.ipynb
notebook/data/label_enrichment.ipynb
notebook/data/llm_review_urgency.ipynb
notebook/baseline/baseline_models.ipynb
notebook/baseline/06_baseline_inference_demo.ipynb
notebook/transformer/train_xlmr_sentiment.ipynb
notebook/transformer/train_phobertv2_sentiment.ipynb
notebook/transformer/train_videberta_sentiment.ipynb
notebook/transformer/train_phobertlarge_sentiment.ipynb
notebook/transformer/train_phobertv2_topic.ipynb
notebook/transformer/train_phobertv2_topic_noweight.ipynb
notebook/transformer/train_phobertv2_topic_sqrt_weight.ipynb
```

## LLM API key

Notebook `llm_review_urgency.ipynb` co the dung OpenAI API de review urgency labels.

Tao file `.env` tu mau:

```text
OPENAI_API_KEY=sk-...
```

Khong commit `.env`. Repo chi commit `.env.example`.

Trong notebook LLM review, mac dinh nen test truoc:

```python
RUN_LLM_REVIEW = True
MAX_REVIEW_ROWS = 30
```

Sau khi ket qua on:

```python
MAX_REVIEW_ROWS = None
```

## Ket qua data hien tai

Sau merge:

```text
Rows: 49,141
Columns: 11
Empty text rows: 0
Duplicate text rows: 1
```

Sau LLM review urgency:

```text
Review candidates: 921
LLM reviewed rows: 921
LLM/rule disagreements: 309
Final urgency:
  low:    48,764
  medium:    335
  high:       42
```

## Baseline results

Best test results hien tai:

| Task | Best model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| sentiment_3class | TF-IDF + Linear SVM | 0.819 | 0.812 | 0.819 |
| topic_group | TF-IDF + Linear SVM | 0.815 | 0.658 | 0.816 |
| toxic_binary | TF-IDF + Linear SVM | 0.992 | 0.901 | 0.991 |
| urgency_final | TF-IDF + Linear SVM | 0.996 | 0.751 | 0.996 |

Luu y:

- Sentiment la task sach nhat.
- Topic_group van lech lop, can doc macro-F1.
- Toxic va urgency co label enrichment/rule/LLM, khong nen chi nhin accuracy.
- Urgency `high` rat it, nen can than khi ket luan.

## Transformer sentiment results

Da fine-tune va danh gia 4 Transformer cho task:

```text
sentiment_std_3class
```

Ket qua test:

| Rank | Model | Accuracy | Macro-F1 | Weighted-F1 | Ghi chu |
|---:|---|---:|---:|---:|---|
| 1 | `vinai/phobert-base-v2` | 0.860 | 0.858 | 0.860 | Model sentiment chinh |
| 2 | `vinai/phobert-large` | 0.855 | 0.853 | 0.855 | Nang hon nhung khong tot hon base-v2 |
| 3 | `FacebookAI/xlm-roberta-base` | 0.855 | 0.852 | 0.855 | Multilingual baseline tot |
| 4 | `Fsoft-AIC/videberta-base` | 0.735 | 0.710 | 0.729 | Khong can uu tien tiep |
| 5 | `TF-IDF + Linear SVM` | 0.819 | 0.812 | 0.819 | Baseline classic |

Ket luan:

```text
vinai/phobert-base-v2
```

la model sentiment tot nhat hien tai, vua co macro-F1 cao nhat vua nhe hon PhoBERT-large.

Bang tong hop sentiment:

```text
outputs/reports/transformer/sentiment_model_comparison.csv
outputs/reports/transformer/sentiment_model_comparison.md
```

Model sentiment tot nhat duoc luu tren Drive tai:

```text
outputs/models/transformer/phobert_base_v2_sentiment_20260620_030200/model
```

## Transformer topic results

Da fine-tune `topic_group` voi `vinai/phobert-base-v2` theo 3 bien the:

| Rank | Model topic | Accuracy | Macro-F1 | Weighted-F1 | Ghi chu |
|---:|---|---:|---:|---:|---|
| 1 | `topic_phobertv2_noweight` | 0.848 | 0.722 | 0.846 | Model topic chinh |
| 2 | `topic_phobertv2_sqrt_weight` | 0.839 | 0.722 | 0.841 | Gan bang no-weight, tot hon cho mot so lop nho |
| 3 | `topic_phobertv2` full weight | 0.830 | 0.716 | 0.835 | Bi class weight keo manh, khong chon lam model chinh |
| 4 | `TF-IDF + Linear SVM` | 0.815 | 0.658 | 0.816 | Baseline classic |

Ket luan:

```text
topic_phobertv2_noweight
```

la model topic chinh hien tai vi co accuracy va weighted-F1 cao nhat, macro-F1 cung nhinh hon `sqrt_weight` mot chut. Ban `sqrt_weight` duoc giu lai nhu mot thuc nghiem tham khao neu muon uu tien them cac lop nho nhu `spam`.

Notebook topic:

```text
notebook/transformer/train_phobertv2_topic.ipynb
notebook/transformer/train_phobertv2_topic_noweight.ipynb
notebook/transformer/train_phobertv2_topic_sqrt_weight.ipynb
```

Report topic:

```text
outputs/reports/transformer/topic_phobertv2/
outputs/reports/transformer/topic_phobertv2_noweight/
outputs/reports/transformer/topic_phobertv2_sqrt_weight/
```

Model topic chinh duoc luu tren Drive tai:

```text
outputs/models/transformer/phobert_base_v2_topic_20260621_101250/model
```

## Reports

Sau khi chay notebook, cac report nam o:

```text
outputs/reports/data/data_merge_report.md
outputs/reports/data/eda_report.md
outputs/reports/data/label_enrichment_report.md
outputs/reports/data/llm_urgency_review_report.md
outputs/reports/baseline/baseline_report.md
outputs/reports/baseline/baseline_results.csv
outputs/reports/transformer/sentiment_model_comparison.csv
outputs/reports/transformer/sentiment_model_comparison.md
outputs/reports/transformer/xlmr/
outputs/reports/transformer/phobertv2/
outputs/reports/transformer/phobert_large/
outputs/reports/transformer/videberta/
outputs/reports/transformer/topic_phobertv2/
outputs/reports/transformer/topic_phobertv2_noweight/
outputs/reports/transformer/topic_phobertv2_sqrt_weight/
```

## Baseline inference demo

Dung notebook:

```text
notebook/baseline/06_baseline_inference_demo.ipynb
```

Notebook nay test nhanh tren danh sach `sample_texts` va tra ve:

- sentiment
- topic
- toxic
- urgency

Neu da chay `baseline_models.ipynb`, demo se load model tu:

```text
outputs/models/baseline/
```

Neu chua co model saved, demo co fallback train lai.

## Ghi chu ve GitHub

Nen push:

- notebooks
- source code
- reports
- `PLAN.md`
- `.gitignore`
- `.env.example`
- `README.md`

Khong nen push:

- `.env`
- dataset CSV
- processed CSV
- model files
- vector DB
- generated figures neu khong can
