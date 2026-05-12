import pandas as pd

# Load the full trend file (with sales)
full = pd.read_csv("daily_trend_signals_with_real_sales.csv", parse_dates=['date'])

# Load the NLP‑enriched file (the one with 21 rows)
nlp = pd.read_csv("daily_trend_signals_with_nlp_yolos.csv", parse_dates=['date'])

# Identify NLP columns (those not in the original trend file)
original_cols = full.columns.tolist()
nlp_cols = [col for col in nlp.columns if col not in original_cols and col != 'date']

# Keep only date and NLP columns from nlp
nlp_part = nlp[['date'] + nlp_cols]

# Outer join (keeps all dates from full)
merged = pd.merge(full, nlp_part, on='date', how='outer')

# Fill missing NLP scores with 0
merged[nlp_cols] = merged[nlp_cols].fillna(0)

# Sort by date
merged = merged.sort_values('date').reset_index(drop=True)

# Save the full dataset
merged.to_csv("daily_trend_signals_with_nlp_full.csv", index=False)

print(f"Full dataset has {len(merged)} rows.")