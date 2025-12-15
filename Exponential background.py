import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

# --------------------------------------------------
# 1. Load Excel File (generalized)
# --------------------------------------------------
df = pd.read_excel("data set.xlsx")   # must contain: X, Y1, Y2, ...

# Ensure X column is named "x"
df.rename(columns={df.columns[0]: "x"}, inplace=True)

# Identify all Y columns automatically
y_columns = [col for col in df.columns if col != "x"]

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("x").reset_index(drop=True)

print("Detected Y columns:", y_columns)

# --------------------------------------------------
# Mask regions (same masks for all Y's)
# --------------------------------------------------
mask_1 = (df["x"] >= 245) & (df["x"] <= 260)
mask_2 = (df["x"] >= 219) & (df["x"] <= 225)
mask = mask_1 | mask_2

x_fit = df.loc[mask, "x"].values

# --------------------------------------------------
# Exponential decay model
# --------------------------------------------------
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# --------------------------------------------------
# Loop over ALL Y columns
# --------------------------------------------------
for y_col in y_columns:
    print("\n==============================")
    print(f"FITTING COLUMN: {y_col}")
    print("==============================")

    y_fit = df.loc[mask, y_col].values

    # Initial guess
    p0 = [y_fit.max() - y_fit.min(), 0.01, y_fit.min()]

    # Fit
    popt, pcov = curve_fit(exp_decay, x_fit, y_fit, p0=p0, maxfev=20000)
    a, b, c = popt

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")

    # Full prediction
    x_all = df["x"].values
    y_pred = exp_decay(x_all, *popt)
    residuals = df[y_col] - y_pred

    # --------------------------------------------------
    # Plot Data + Fit
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df["x"], df[y_col], 'o', label=f"Data ({y_col})", markersize=4)
    x_grid = np.linspace(df["x"].min(), df["x"].max(), 500)
    plt.plot(x_grid, exp_decay(x_grid, *popt), 'r-', label="Exponential fit", linewidth=2)
    plt.xlabel("x")
    plt.ylabel(y_col)
    plt.title(f"Exponential Fit for {y_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Residual Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(df["x"], residuals, 'o-', markersize=4)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("x")
    plt.ylabel(f"Residual ({y_col})")
    plt.title(f"Residual Plot for {y_col}")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Save prediction CSV for this Y column
    # --------------------------------------------------
    output = pd.DataFrame({
        "x": df["x"],
        f"{y_col}_real": df[y_col],
        f"{y_col}_predicted": y_pred,
        f"{y_col}_residual": residuals
    })

    fname = f"prediction_{y_col}.csv"
    output.to_csv(fname, index=False)
    print(f"Saved: {fname}")

# ==================================================================
# 8. MERGE ALL RESIDUAL COLUMNS INTO ONE XLSX FILE
# ==================================================================

print("\nMerging all residuals into one file...")

# Find all prediction CSVs
files = sorted(glob.glob("prediction_*.csv"))

merged_df = None

for file in files:
    df_pred = pd.read_csv(file)

    # Identify residual column
    residual_col = [col for col in df_pred.columns if "residual" in col.lower()][0]

    # Start merged table with X column
    if merged_df is None:
        merged_df = df_pred[["x", residual_col]].copy()
    else:
        merged_df = merged_df.merge(df_pred[["x", residual_col]], on="x", how="outer")

# Save merged residuals
merged_df.to_excel("all_residuals.xlsx", index=False)

print("Saved merged residuals to all_residuals.xlsx")
