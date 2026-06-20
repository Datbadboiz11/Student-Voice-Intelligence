# Data Merge Report

## Output

- CSV: `data\processed\student_voice_merged.csv`
- Rows: `49141`
- Columns: `11`

## Input files

- `datasets\NEU-ESC\train_set.csv`: 23048 rows, 3 columns
- `datasets\NEU-ESC\val_set.csv`: 3305 rows, 3 columns
- `datasets\NEU-ESC\test_set.csv`: 6613 rows, 3 columns
- `datasets\UIT-VSFC\uit_vsfc_train.csv`: 11426 rows, 3 columns
- `datasets\UIT-VSFC\uit_vsfc_validation.csv`: 1583 rows, 3 columns
- `datasets\UIT-VSFC\uit_vsfc_test.csv`: 3166 rows, 3 columns

## Schema

- `text`
- `source_dataset`
- `split_original`
- `sentiment_raw`
- `sentiment_std_code`
- `sentiment_std_3class`
- `sentiment_std_4class`
- `topic_raw`
- `topic_std`
- `is_toxic`
- `urgency_level`

## Quality checks

- Empty text rows removed: `0`
- Duplicate text rows reported, not dropped: `1`

## Distributions

- Rows by source: `{'NEU_ESC': 32966, 'UIT_VSFC': 16175}`
- Rows by split: `{'train': 34474, 'test': 9779, 'validation': 4888}`
- Sentiment std code: `{0: 23471, 2: 12639, 1: 12186, 3: 845}`
- Sentiment 3-class: `{'neutral': 23471, 'negative': 13484, 'positive': 12186}`
- Sentiment 4-class: `{'neutral': 23471, 'negative': 12639, 'positive': 12186, 'toxic': 845}`
- Topic: `{'others': 15218, 'lecturer': 11607, 'academic': 10512, 'training_program': 3040, 'student_services': 2358, 'personal_affairs': 1478, 'news': 902, 'jobs_recruitment': 808, 'social_affairs': 769, 'facilities': 712, 'help_share': 670, 'clubs_events': 662, 'spam': 405}`
- Toxic: `{0: 48296, 1: 845}`
- Urgency: `{'low': 49141}`

## Mapping notes

- `sentiment_std_code` dung chung quy uoc: `0=neutral`, `1=positive`, `2=negative`, `3=toxic`.
- NEU raw sentiment da trung quy uoc chuan: `0=neutral`, `1=positive`, `2=negative`, `3=toxic`.
- UIT raw sentiment duoc doi code truoc khi gop: `0=negative -> 2`, `1=neutral -> 0`, `2=positive -> 1`.
- NEU `sentiment_std_code=3` duoc map thanh `negative` trong `sentiment_std_3class`, thanh `toxic` trong `sentiment_std_4class`, va `is_toxic=1`.
- UIT khong co nhan toxic rieng, nen `is_toxic=0` cho UIT.
- `urgency_level` tam thoi la `low` cho toan bo du lieu. Phase sau co the thay bang rule-based labeling.
- `topic_std` hien la nhan text da chuan hoa tu nhan so cua tung dataset; chua ep NEU va UIT ve mot taxonomy hep de tranh mapping sai.