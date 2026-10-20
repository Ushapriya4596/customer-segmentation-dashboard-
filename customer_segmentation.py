"""
Customer Segmentation using K-Means
Beginner-friendly project script.

Run from the project folder:
    python customer_segmentation.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------- 1. File locations and settings --------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "customers.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Five clusters is a starter choice. Use the elbow chart to experiment.
K = 5
FEATURES = ["Age", "AnnualIncome_kUSD", "SpendingScore", "AnnualPurchases"]

sns.set_theme(style="whitegrid")


def main():
    # -------------------- 2. Load and inspect data --------------------
    df = pd.read_csv(DATA_PATH)
    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string(index=False))
    print("\n=== DATASET SHAPE ===")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().to_string())

    # Drop exact duplicate rows and rows missing required features.
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=FEATURES).copy()

    # Keep only valid, non-negative numeric values for model inputs.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).copy()
    df = df[(df["Age"] > 0) & (df["AnnualIncome_kUSD"] >= 0)
            & (df["SpendingScore"].between(1, 100))
            & (df["AnnualPurchases"] >= 0)].copy()

    if len(df) < 6:
        raise ValueError("At least 6 valid customer rows are needed for clustering.")

    # -------------------- 3. Scale features --------------------
    # Scaling stops large-number features from dominating distance calculations.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # -------------------- 4. Elbow Method --------------------
    inertias = []
    k_values = range(2, min(10, len(df) - 1) + 1)
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("Elbow Method: Choosing the Number of Clusters")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_elbow_method.png", dpi=160)
    plt.close()

    # -------------------- 5. Fit final K-Means model --------------------
    k_final = min(K, len(df) - 1)
    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df["Cluster"]) if df["Cluster"].nunique() > 1 else float("nan")
    print(f"\nSilhouette score for K={k_final}: {score:.3f}")
    print("(A higher silhouette score can indicate better-separated clusters; use it with charts and business context.)")

    # -------------------- 6. Save cluster results and summary --------------------
    df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    summary = df.groupby("Cluster").agg(
        CustomerCount=("CustomerID", "count"),
        AverageAge=("Age", "mean"),
        AverageAnnualIncome_kUSD=("AnnualIncome_kUSD", "mean"),
        AverageSpendingScore=("SpendingScore", "mean"),
        AverageAnnualPurchases=("AnnualPurchases", "mean"),
    ).round(2)
    summary.to_csv(OUTPUT_DIR / "segment_summary.csv")

    print("\n=== SEGMENT SUMMARY ===")
    print(summary.to_string())

    # -------------------- 7. Graph: income vs spending --------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="AnnualIncome_kUSD", y="SpendingScore",
        hue="Cluster", palette="tab10", s=75, alpha=0.85
    )
    plt.title("Customer Segments: Annual Income vs Spending Score")
    plt.xlabel("Annual income (thousands of USD)")
    plt.ylabel("Spending score (1–100)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_customer_clusters.png", dpi=160)
    plt.close()

    # -------------------- 8. Graph: segment profiles --------------------
    profile = summary[[
        "AverageAge", "AverageAnnualIncome_kUSD",
        "AverageSpendingScore", "AverageAnnualPurchases"
    ]].copy()
    # Standardize summary columns for a comparable heatmap view.
    profile_z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    plt.figure(figsize=(10, 5))
    sns.heatmap(profile_z, annot=True, fmt=".1f", cmap="vlag", center=0)
    plt.title("Segment Profiles (relative to overall segment averages)")
    plt.xlabel("Customer characteristic")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_segment_profile_heatmap.png", dpi=160)
    plt.close()

    # -------------------- 9. Graph: number of customers in each segment --------------------
    plt.figure(figsize=(8, 5))
    counts = df["Cluster"].value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Number of Customers in Each Segment")
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_segment_sizes.png", dpi=160)
    plt.close()

    # -------------------- 10. Simple business interpretation --------------------
    overall_spend = df["SpendingScore"].mean()
    overall_income = df["AnnualIncome_kUSD"].mean()
    print("\n=== SUGGESTED INTERPRETATIONS ===")
    for cluster_id, row in summary.iterrows():
        tags = []
        if row["AverageSpendingScore"] >= overall_spend + 8:
            tags.append("higher-spending")
        elif row["AverageSpendingScore"] <= overall_spend - 8:
            tags.append("lower-spending")
        else:
            tags.append("moderate-spending")

        if row["AverageAnnualIncome_kUSD"] >= overall_income + 8:
            tags.append("higher-income")
        elif row["AverageAnnualIncome_kUSD"] <= overall_income - 8:
            tags.append("lower-income")
        else:
            tags.append("mid-range-income")

        print(f"\nCluster {cluster_id}: {', '.join(tags)}; "
              f"{int(row['CustomerCount'])} customers.")
        if "higher-spending" in tags:
            print("  Possible action: loyalty rewards or early access to relevant products.")
        elif "lower-spending" in tags:
            print("  Possible action: test useful bundles or targeted offers; avoid assuming price is the only reason.")
        else:
            print("  Possible action: personalized recommendations and measure response.")

    print("\nDone! Check the outputs folder for CSV files and four PNG graphs.")
    print("Note: Cluster numbers are arbitrary labels; Cluster 0 is not automatically better than Cluster 1.")


if __name__ == "__main__":
    main()
