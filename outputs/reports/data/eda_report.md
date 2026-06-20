# EDA Report

## Input

- CSV: `data\processed\student_voice_merged.csv`
- Rows: `49141`
- Columns: `11`

## Data quality

- Duplicate text rows: `1`
- Duplicate text groups across splits: `0`

## Missing values

|                      |   missing_count |   missing_pct |
|:---------------------|----------------:|--------------:|
| text                 |               0 |             0 |
| source_dataset       |               0 |             0 |
| split_original       |               0 |             0 |
| sentiment_raw        |               0 |             0 |
| sentiment_std_code   |               0 |             0 |
| sentiment_std_3class |               0 |             0 |
| sentiment_std_4class |               0 |             0 |
| topic_raw            |               0 |             0 |
| topic_std            |               0 |             0 |
| is_toxic             |               0 |             0 |
| urgency_level        |               0 |             0 |

## Source distribution

| source_dataset   |   count |   pct |
|:-----------------|--------:|------:|
| NEU_ESC          |   32966 | 67.08 |
| UIT_VSFC         |   16175 | 32.92 |

## Split distribution

| split_original   |   count |   pct |
|:-----------------|--------:|------:|
| train            |   34474 | 70.15 |
| test             |    9779 | 19.9  |
| validation       |    4888 |  9.95 |

## Sentiment 3-class distribution

| sentiment_std_3class   |   count |   pct |
|:-----------------------|--------:|------:|
| neutral                |   23471 | 47.76 |
| negative               |   13484 | 27.44 |
| positive               |   12186 | 24.8  |

## Sentiment 4-class distribution

| sentiment_std_4class   |   count |   pct |
|:-----------------------|--------:|------:|
| neutral                |   23471 | 47.76 |
| negative               |   12639 | 25.72 |
| positive               |   12186 | 24.8  |
| toxic                  |     845 |  1.72 |

## Topic distribution

| topic_std        |   count |   pct |
|:-----------------|--------:|------:|
| others           |   15218 | 30.97 |
| lecturer         |   11607 | 23.62 |
| academic         |   10512 | 21.39 |
| training_program |    3040 |  6.19 |
| student_services |    2358 |  4.8  |
| personal_affairs |    1478 |  3.01 |
| news             |     902 |  1.84 |
| jobs_recruitment |     808 |  1.64 |
| social_affairs   |     769 |  1.56 |
| facilities       |     712 |  1.45 |
| help_share       |     670 |  1.36 |
| clubs_events     |     662 |  1.35 |
| spam             |     405 |  0.82 |

## Text length summary

|       |   char_count |   word_count |
|:------|-------------:|-------------:|
| count |     49141    |     49141    |
| mean  |        89.57 |        21.51 |
| std   |       180.29 |        41.19 |
| min   |         3    |         1    |
| 25%   |        31    |         8    |
| 50%   |        48    |        12    |
| 75%   |        85    |        20    |
| 90%   |       166    |        40    |
| 95%   |       269    |        63    |
| 99%   |       747    |       171    |
| max   |      6815    |      1578    |

## Figures

- `outputs\figures\source_distribution.png`
- `outputs\figures\split_distribution.png`
- `outputs\figures\sentiment_std_code_distribution.png`
- `outputs\figures\sentiment_std_3class_distribution.png`
- `outputs\figures\sentiment_std_4class_distribution.png`
- `outputs\figures\topic_distribution.png`
- `outputs\figures\toxic_distribution.png`
- `outputs\figures\urgency_distribution.png`
- `outputs\figures\text_length_distribution.png`
- `outputs\figures\sentiment_topic_heatmap.png`
- `outputs\figures\negative_feedback_top_words.png`

## Notes

- Neu `sentiment_std_code` thieu, hay rerun Phase 1 de cap nhat schema moi nhat.
- Duplicate hien chi duoc bao cao, chua drop tu dong.
- Topic cua NEU va UIT khong hoan toan cung taxonomy, can can than khi train mot topic classifier chung.
- Khi label lech manh, baseline nen bao cao macro-F1 va weighted-F1, khong chi accuracy.