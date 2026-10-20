
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")
st.caption("Explore customer groups, purchasing behavior, and spending patterns.")

# Find CSV files whether they are in the root folder or subfolders.
def find_csv(filename, alternatives=None):
    alternatives = alternatives or []
    paths = [filename] + alternatives

    for path in paths:
        if Path(path).is_file():
            return pd.read_csv(path)

    return None


# Load customer data
customers = find_csv(
    "customers.csv",
    ["data/customers.csv"]
)

segments = find_csv(
    "customer_segments.csv",
    ["outputs/customer_segments.csv"]
)

summary = find_csv(
    "segment_summary.csv",
    ["outputs/segment_summary.csv"]
)

if customers is None and segments is None:
    st.error(
        "Customer data was not found. "
        "Please check that customers.csv or customer_segments.csv "
        "is uploaded to your GitHub repository."
    )
    st.stop()

# Prefer segmented data if it already contains cluster assignments.
if segments is not None:
    df = segments.copy()
else:
    df = customers.copy()

# Standardize common column names.
rename_map = {
    "AnnualIncome_kUSD": "AnnualIncome",
    "Annual Income (k$)": "AnnualIncome",
    "Spending Score (1-100)": "SpendingScore",
    "Annual Purchases": "AnnualPurchases",
    "Cluster": "Segment"
}
df = df.rename(columns=rename_map)

# If no segment labels exist, explain that clustering is not included.
segment_column = next(
    (col for col in ["Segment", "Cluster", "cluster", "Labels"]
     if col in df.columns),
    None
)

if segment_column is None:
    st.info(
        "Your customer dataset is loaded, but it has no cluster labels yet. "
        "Run the clustering script to generate customer segments."
    )

# Numeric columns used for analysis.
numeric_columns = df.select_dtypes(include="number").columns.tolist()

# Sidebar filters
st.sidebar.header("Dashboard Filters")
if segment_column:
    segment_values = sorted(df[segment_column].dropna().unique().tolist(),
                            key=str)
    selected_segments = st.sidebar.multiselect(
        "Select customer segments",
        options=segment_values,
        default=segment_values
    )
    filtered = df[df[segment_column].isin(selected_segments)].copy()
else:
    filtered = df.copy()

# KPI cards
c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(filtered):,}")

if "SpendingScore" in filtered.columns:
    c2.metric(
        "Average Spending Score",
        f"{filtered['SpendingScore'].mean():.1f}"
    )
else:
    c2.metric("Average Spending Score", "N/A")

if "AnnualIncome" in filtered.columns:
    c3.metric(
        "Average Annual Income",
        f"{filtered['AnnualIncome'].mean():,.1f}"
    )
else:
    c3.metric("Average Annual Income", "N/A")

st.divider()

# Charts
if segment_column:
    st.subheader("Customer Segment Distribution")
    counts = filtered[segment_column].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig = px.bar(
        counts,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Number of Customers in Each Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    if "AnnualIncome" in filtered.columns and "SpendingScore" in filtered.columns:
        st.subheader("Income vs Spending")
        color_col = segment_column if segment_column else None
        fig = px.scatter(
            filtered,
            x="AnnualIncome",
            y="SpendingScore",
            color=color_col,
            hover_data=[
                col for col in ["CustomerID", "Age", "AnnualPurchases"]
                if col in filtered.columns
            ],
            title="Customer Income and Spending Patterns"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "Age" in filtered.columns:
        st.subheader("Customer Age Distribution")
        fig = px.histogram(filtered, x="Age", title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

with right:
    if "Age" in filtered.columns:
        st.subheader("Age Distribution")
        fig = px.histogram(
            filtered,
            x="Age",
            color=segment_column if segment_column else None,
            title="Customer Ages"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "AnnualPurchases" in filtered.columns:
        st.subheader("Annual Purchases")
        fig = px.histogram(
            filtered,
            x="AnnualPurchases",
            title="Annual Purchases Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# Segment summary table
if summary is not None:
    st.subheader("Segment Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

# Data preview and download
with st.expander("View customer data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered customer data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_customer_segments.csv",
    mime="text/csv"
)

st.success("Customer Segmentation Dashboard loaded successfully.")
