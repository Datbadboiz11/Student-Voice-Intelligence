# LLM Urgency Review Report

## Files

- Input: `data\processed\student_voice_enriched.csv`
- Review candidates: `data\processed\urgency_review_candidates.csv`
- LLM checkpoint: `data\processed\urgency_llm_review.csv`
- Output: `data\processed\student_voice_enriched_reviewed.csv`

## Counts

- Review candidates: `921`
- LLM reviewed rows: `921`
- LLM/rule disagreements: `309`

## Distributions

- Rule urgency: `{'low': 48502, 'medium': 445, 'high': 194}`
- Final urgency: `{'low': 48764, 'medium': 335, 'high': 42}`
- Final source: `{'rule': 48220, 'llm': 921}`

## Notes

- `urgency_level` la nhan rule-based tu notebook 03.
- `urgency_level_llm` la nhan LLM review neu da chay API.
- `urgency_level_final` uu tien LLM label, fallback ve rule label neu chua review.
- Nen human-check ngau nhien 100-200 mau da LLM review truoc khi train chinh thuc.