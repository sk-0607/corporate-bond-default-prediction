"""
demo_src.py  —  Shows output from every src module using real project data.
Run from project root:  python demo_src.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info messages

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. src.data.load_data — load_processed()")
print("="*60)
from src.data.load_data import load_processed

train_df, test_df = load_processed(
    "data/processed/features_train.csv",
    "data/processed/features_test.csv",
)
print(f"  Train shape : {train_df.shape}")
print(f"  Test  shape : {test_df.shape}")
print(f"  Columns     : {list(train_df.columns)}")

# ── 2. Class weights ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. src.data.preprocessing — compute_class_weights()")
print("="*60)
from src.data.preprocessing import compute_class_weights

cw = compute_class_weights(train_df["default"])
print(f"  Class weights : {cw}")

# ── 3. Feature engineering ────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. src.features.build_features — engineer_features()")
print("="*60)
from src.features.build_features import engineer_features, COLS_TO_DROP

train_eng, test_eng = engineer_features(train_df, test_df)
print(f"  Dropped cols  : {COLS_TO_DROP}")
print(f"  Train shape   : {train_eng.shape}")
print(f"  Test  shape   : {test_eng.shape}")
print(f"  Train default rate : {train_eng['default'].mean():.2%}")
print(f"  Test  default rate : {test_eng['default'].mean():.2%}")

# ── 4. Sequence creation ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. src.features.build_features — create_sequences()")
print("="*60)
from src.features.build_features import create_sequences

X_train, y_train = create_sequences(train_eng, sequence_length=4)
X_test,  y_test  = create_sequences(test_eng,  sequence_length=4)
print(f"  X_train : {X_train.shape}  (sequences, timesteps, features)")
print(f"  X_test  : {X_test.shape}")
print(f"  Train default rate in sequences : {y_train.mean():.2%}")
print(f"  Test  default rate in sequences : {y_test.mean():.2%}")

# ── 5. Build LSTM & GRU ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. src.models — build_lstm_model() / build_gru_model()")
print("="*60)
from src.models.train_lstm import build_lstm_model
from src.models.train_gru  import build_gru_model

input_shape = (X_train.shape[1], X_train.shape[2])
lstm = build_lstm_model(input_shape)
gru  = build_gru_model(input_shape)
print(f"  LSTM total params : {lstm.count_params():,}")
print(f"  GRU  total params : {gru.count_params():,}")

# ── 6. Threshold helpers ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. src.utils.helpers — evaluate_model() on dummy data")
print("="*60)
import numpy as np
from src.utils.helpers import evaluate_model, find_threshold_f1, find_threshold_cost

# Use dummy proba to show evaluate_model output shape
rng    = np.random.default_rng(42)
y_true = (rng.random(200) > 0.9).astype(int)   # ~10% positives
y_prob = rng.beta(0.5, 5, size=200)
y_pred = (y_prob >= 0.5).astype(int)

metrics  = evaluate_model(y_true, y_pred, y_prob, dataset_name="Demo")
t_f1     = find_threshold_f1(y_true, y_prob)
t_cost, min_cost = find_threshold_cost(y_true, y_prob, fn_cost=10, fp_cost=1)
print(f"\n  F1-optimal threshold   : {t_f1:.3f}")
print(f"  Cost-optimal threshold : {t_cost:.3f}  (min cost = {min_cost})")

print("\n" + "="*60)
print("All src modules executed successfully.")
print("="*60 + "\n")
