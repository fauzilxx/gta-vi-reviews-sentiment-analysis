# Rating-Proxy vs. Human Sentiment Alignment (60-Row Training Sample)

- **Sample size:** 60 reviews (20 per rating bucket from training split)
- **Exact agreement:** 45/60 (75.0%)
- **Cohen's kappa:** 0.6250

## Per-Bucket Alignment

| Rating Bucket | Sampled | Matches | Alignment | Human Label Breakdown |
|---|---:|---:|---:|---|
| Negative | 20 | 18 | 90.0% | Negative: 18, Neutral: 2 |
| Neutral | 20 | 7 | 35.0% | Negative: 12, Neutral: 7, Positive: 1 |
| Positive | 20 | 20 | 100.0% | Positive: 20 |

## Confusion Matrix (Rows = Rating Proxy, Columns = Human Sentiment)

| Proxy \\ Human | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 18 | 2 | 0 |
| Neutral | 12 | 7 | 1 |
| Positive | 0 | 0 | 20 |

## Key Finding

- **Positive (8–10)** and **Negative (0–4)** align strongly with human text sentiment (100% and 90%).
- **Neutral (5–7)** aligns only **35%** of the time; most 5–7 reviews express predominantly negative criticism in text despite a mid-range numeric score.
