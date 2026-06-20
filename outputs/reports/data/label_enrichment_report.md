# Label Enrichment Report

## Output

- Input CSV: `data\processed\student_voice_merged.csv`
- Output CSV: `data\processed\student_voice_enriched.csv`
- Rows: `49141`
- Columns: `19`

## Toxic

- Toxic distribution: `{0: 48009, 1: 1132}`
- Toxic label source: `{'default_non_toxic': 48009, 'seed_and_rule': 497, 'seed_neu': 348, 'rule': 287}`

## Urgency

- Urgency distribution: `{'low': 48502, 'medium': 445, 'high': 194}`
- Urgency label source: `{'default_low': 48502, 'rule_medium': 445, 'rule_high': 194}`

## Topic group

- Topic group distribution: `{'teaching_learning': 25159, 'others': 15218, 'student_services': 3028, 'personal_social': 2247, 'events_news': 1564, 'career_jobs': 808, 'facilities': 712, 'spam': 405}`
- Label needs review: `{0: 48220, 1: 921}`

## Notes

- `is_toxic_seed` giu toxic goc tu Phase 1 truoc khi mo rong bang rule.
- `is_toxic_rule` la nhan toxic do keyword/rule bat duoc.
- `is_toxic` sau enrichment la hop cua seed va rule.
- `urgency_level` la rule-based label, can review thu cong neu dung de train model nghiem tuc.
- `topic_group` la taxonomy tho, khong thay the `topic_std` goc.