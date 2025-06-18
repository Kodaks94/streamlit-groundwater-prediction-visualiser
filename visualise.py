import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
true_values = np.load("results/true_values.npy")          # Flattened ground truth
memo_preds = np.load("results/memo_preds.npy")            # Flattened memoformer predictions
xgb_preds = np.load("results/xgb_preds.npy")              # One per sequence
rf_preds = np.load("results/rf_preds.npy")
gb_preds = np.load("results/gb_preds.npy")
mlp_preds = np.load("results/mlp_preds.npy")

# === Determine repeat factor ===
# Assuming true_values and memo_preds are flattened (num_seq * pred_len)
# and xgb_preds etc. are (num_seq,)
repeat_factor = int(len(true_values) / len(xgb_preds))  # Should be pred_len

# === Repeat other model predictions to match timeline ===
xgb_expanded = np.repeat(xgb_preds, repeat_factor)
rf_expanded = np.repeat(rf_preds, repeat_factor)
gb_expanded = np.repeat(gb_preds, repeat_factor)
mlp_expanded = np.repeat(mlp_preds, repeat_factor)

# === Streamlit app ===
st.title("Groundwater Level Forecast Model Comparison")

st.sidebar.title("Select Models to Display")
show_true = st.sidebar.checkbox("Ground Truth", value=True)
show_memo = st.sidebar.checkbox("MemoFormer", value=True)
show_xgb = st.sidebar.checkbox("XGBoost", value=False)
show_rf = st.sidebar.checkbox("Random Forest", value=False)
show_gb = st.sidebar.checkbox("Gradient Boosting", value=False)
show_mlp = st.sidebar.checkbox("MLP", value=False)

# === Plot ===
st.subheader("Forecast comparison")

fig, ax = plt.subplots(figsize=(14, 6))
window_size = st.sidebar.slider("Smoothing window size", 1, 50, 10)

def smooth(y, window_size=10):
    y = y.flatten()  # Ensure 1D
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')


if show_true:
    ax.plot(smooth(true_values), label="Ground Truth", linewidth=0.8, color="black")
if show_memo:
    ax.plot(smooth(memo_preds), label="MemoFormer", linewidth=1.2, alpha=0.7)
if show_xgb:
    ax.plot(smooth(xgb_expanded), label="XGBoost", linewidth=1.2, alpha=0.7)
if show_rf:
    ax.plot(smooth(rf_expanded), label="Random Forest", linewidth=1.2, alpha=0.7)
if show_gb:
    ax.plot(smooth(gb_expanded), label="Gradient Boosting", linewidth=1.2, alpha=0.7)
if show_mlp:
    ax.plot(smooth(mlp_expanded), label="MLP", linewidth=1.2, alpha=0.7)

ax.set_title("Model Predictions vs. Ground Truth")
ax.set_xlabel("Time steps (flattened prediction windows)")
ax.set_ylabel("Groundwater Level (ft)")
ax.grid(True)
ax.legend()

st.pyplot(fig)
