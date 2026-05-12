"""
train_tft_with_nlp.py
---------------------
Trains a Temporal Fusion Transformer (TFT) on the prepared multi‑modal dataset
(Google Trends, visual counts, NLP scores, and real sales).

Input:  data/tft_data_with_nlp.csv   (produced by prepare_tft_data_with_nlp.py)
Output: data/tft_model_nlp.ckpt      (trained model checkpoint)
"""

import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import os

# ------------------------------------------------------------
# 1. Paths – data in ../data/
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

INPUT_CSV = os.path.join(DATA_DIR, 'tft_data_with_nlp.csv')
MODEL_CHECKPOINT = os.path.join(DATA_DIR, 'tft_model_nlp.ckpt')

# ------------------------------------------------------------
# 2. Load the prepared data and rename 'date' to 'time'
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
df = df.rename(columns={'date': 'time'})       # TFT expects a column named 'time'

# ------------------------------------------------------------
# 3. Set forecast horizons
# ------------------------------------------------------------
max_encoder_length = 30        # use last 30 days to predict
max_prediction_length = 30     # forecast next 30 days

# ------------------------------------------------------------
# 4. Compute training cutoff (last 30 days used for validation)
# ------------------------------------------------------------
training_cutoff = df['time_idx'].max() - max_prediction_length
print(f"Training cutoff: time_idx <= {training_cutoff}")

train_df = df[df['time_idx'] <= training_cutoff].reset_index(drop=True)
val_df = df[df['time_idx'] > training_cutoff].reset_index(drop=True)

print(f"Training rows: {len(train_df)}, Validation rows: {len(val_df)}")

# ------------------------------------------------------------
# 5. Define the TimeSeriesDataSet
# ------------------------------------------------------------
training = TimeSeriesDataSet(
    train_df,
    time_idx='time_idx',
    target='sales',
    group_ids=['group_id'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=['group_id'],
    time_varying_known_reals=['time_idx'],
    time_varying_unknown_reals=[c for c in df.columns if c not in
                                ['time', 'group_id', 'sales', 'time_idx']],
    target_normalizer=GroupNormalizer(groups=['group_id'], transformation='softplus'),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# ------------------------------------------------------------
# 6. Create dataloaders
# ------------------------------------------------------------
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = training.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# ------------------------------------------------------------
# 7. Set up PyTorch Lightning trainer
# ------------------------------------------------------------
pl.seed_everything(42)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator='cpu',
    devices=1,
)

# ------------------------------------------------------------
# 8. Create the TFT model
# ------------------------------------------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,               # produces 7 quantiles
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# ------------------------------------------------------------
# 9. Train
# ------------------------------------------------------------
print("Starting training...")
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# ------------------------------------------------------------
# 10. Save the best model checkpoint
# ------------------------------------------------------------
trainer.save_checkpoint(MODEL_CHECKPOINT)
print(f"Model saved as {MODEL_CHECKPOINT}")