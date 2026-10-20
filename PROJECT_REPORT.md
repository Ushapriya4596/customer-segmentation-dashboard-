# Customer Segmentation using K-Means — Project Report

## 1. Abstract
This project applies unsupervised machine learning to group customers with similar demographic and purchasing characteristics. K-Means clustering is used to identify groups from age, annual income, spending score, and annual purchase frequency. Visualizations and a segment summary help translate model output into possible marketing actions.

## 2. Problem statement
A business may have many customers but lack a clear view of different shopping patterns. Treating every customer the same can make promotions less relevant. The project explores whether customer records can be grouped into meaningful segments.

## 3. Objectives
- Inspect and clean a customer dataset.
- Scale numeric features before distance-based clustering.
- Use the Elbow Method to examine candidate cluster counts.
- Apply K-Means and assign each record a cluster.
- Visualize clusters and compare their average characteristics.
- Suggest marketing experiments based on observed patterns.

## 4. Dataset and features
The included CSV is a synthetic learning dataset with 320 fictional customers. Fields:
- `CustomerID`: unique identifier, not used to train the model.
- `Age`: customer age in years.
- `AnnualIncome_kUSD`: annual income in thousands of US dollars.
- `SpendingScore`: score from 1 to 100 used as a proxy for spending behavior.
- `AnnualPurchases`: approximate number of purchases in a year.

Because the data is synthetic, its patterns are for demonstration only and should not be presented as findings about real customers.

## 5. Methodology
1. Load the CSV with Pandas.
2. Check data shape, missing values, duplicates, and valid ranges.
3. Standardize the four numeric model features using `StandardScaler`.
4. Fit K-Means models for several K values and save an Elbow Method plot.
5. Fit a final K-Means model with K=5 as a starter configuration.
6. Save each customer's cluster and a cluster-level summary.
7. Create scatter, heatmap, and cluster-size charts.
8. Interpret clusters cautiously and propose marketing tests.

## 6. Why K-Means?
K-Means is a common unsupervised learning algorithm. It assigns each row to one of K groups based on distance to cluster centers. It is appropriate for a first segmentation exercise with numeric features. Feature scaling is important because age, income, spending score, and purchase frequency use different ranges.

## 7. Outputs
Running the script creates:
- `01_elbow_method.png`
- `02_customer_clusters.png`
- `03_segment_profile_heatmap.png`
- `04_segment_sizes.png`
- `customer_segments.csv`
- `segment_summary.csv`

## 8. Evaluation and limitations
The Elbow Method gives a visual clue rather than a guaranteed best K. The script also prints a silhouette score; neither metric replaces domain knowledge. Cluster IDs are arbitrary. Results can change if the features, scaling, or K change. Real deployments should use real, permissioned, representative data, protect customer privacy, test recommendations, and monitor whether actions improve outcomes.

## 9. Conclusion
The project demonstrates an end-to-end customer analytics workflow: data preparation, feature scaling, clustering, visualization, and interpretation. It gives a foundation for more advanced work such as RFM analysis, purchase-category segmentation, and interactive dashboards.
