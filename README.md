# Student Voice Intelligence

He thong NLP phan tich phan hoi sinh vien tieng Viet. Project hien tai tap trung vao pipeline data, EDA, label enrichment va baseline classification truoc khi chuyen sang Transformer/RAG.

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

Dang lam tiep:

- Phase 4: Transformer single-task models
- Phase 5+: multi-task learning, semantic search, RAG, dashboard

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

## Thu tu chay notebook

Chay theo thu tu:

```text
notebook/data/data_merge.ipynb
notebook/data/eda.ipynb
notebook/data/label_enrichment.ipynb
notebook/data/llm_review_urgency.ipynb
notebook/baseline/baseline_models.ipynb
notebook/baseline/06_baseline_inference_demo.ipynb
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

## Reports

Sau khi chay notebook, cac report nam o:

```text
outputs/reports/data_merge_report.md
outputs/reports/eda_report.md
outputs/reports/label_enrichment_report.md
outputs/reports/llm_urgency_review_report.md
outputs/reports/baseline_report.md
outputs/reports/baseline_results.csv
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

