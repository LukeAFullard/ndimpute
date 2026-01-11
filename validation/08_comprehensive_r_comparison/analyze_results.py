import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze():
    # Read the V2 results from the latest run
    df = pd.read_csv("suite_results_v2.csv")

    # 1. Summary Table
    summary = df.groupby(['Dist', 'Limit'])['MAE'].mean().reset_index()
    print("\n--- Mean MAE by Scenario (Corrected) ---")
    print(summary)

    # 2. Heatmap: MAE by N vs Censoring Level (for Lognormal Single)
    def extract_params(row):
        parts = row['Scenario'].split('_')
        n = int(parts[2].replace('n', ''))
        cens = float(parts[3].replace('cens', ''))
        return pd.Series({'N': n, 'Censoring': cens})

    df[['N', 'Censoring']] = df.apply(extract_params, axis=1)

    # Heatmap Data (Lognormal Single)
    heatmap_data = df[(df['Dist'] == 'lognormal') & (df['Limit'] == 'single')].pivot_table(
        index='Censoring', columns='N', values='MAE', aggfunc='mean'
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".4f")
    plt.title("Mean MAE: Lognormal Distribution")
    plt.ylabel("Censoring Level")
    plt.xlabel("Sample Size (N)")
    plt.savefig("heatmap_mae.png")
    print("Saved heatmap_mae.png")

    # 3. Boxplot: Bias by Limit Type
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Limit', y='MAE', hue='Dist')
    plt.title("Distribution of Errors by Limit Type")
    plt.savefig("boxplot_mae.png")
    print("Saved boxplot_mae.png")

if __name__ == "__main__":
    analyze()
