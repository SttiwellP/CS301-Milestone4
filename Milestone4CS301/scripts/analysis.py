# analysis.py
# Full Docker-Ready Script
# Urban Data Science Project

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting project analysis...\n")

air_quality = pd.read_csv(
    "data/Urban Air Quality and Health Impact Dataset.csv"
)

disaster = pd.read_csv(
    "data/us_disaster_declarations.csv"
)

print("Datasets loaded successfully.\n")

print("AIR QUALITY DATASET")
print(air_quality.head())
print(air_quality.info())
print(air_quality.describe())

print("\nDISASTER DATASET")
print(disaster.head())
print(disaster.info())
print(disaster.describe())

print("\nRunning Exploratory Data Analysis...\n")

numerical_cols = [
    "temp",
    "humidity",
    "precip",
    "windspeed",
    "Heat_Index",
    "Severity_Score",
    "Health_Risk_Score"
]

air_quality[numerical_cols].hist(
    bins=20,
    figsize=(14, 10)
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "air_quality_feature_distributions.png"))
plt.clf()

plt.figure(figsize=(12, 8))

sns.heatmap(
    air_quality[numerical_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix - Air Quality Dataset")
plt.savefig(os.path.join(OUTPUT_DIR, "air_quality_correlation_matrix.png"))
plt.clf()

sns.scatterplot(
    x="Severity_Score",
    y="Health_Risk_Score",
    data=air_quality
)

plt.title("Severity Score vs Health Risk Score")
plt.savefig(os.path.join(OUTPUT_DIR, "severity_vs_health_risk.png"))
plt.clf()

print("Running Disaster Dataset EDA...\n")

disaster["declaration_date"] = pd.to_datetime(
    disaster["declaration_date"]
)

disaster["year"] = disaster["declaration_date"].dt.year
disaster["month"] = disaster["declaration_date"].dt.month

sns.histplot(
    disaster["year"],
    bins=30,
    kde=True
)

plt.title("Distribution of Natural Disasters by Year")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "disaster_distribution_by_year.png"))
plt.clf()

top_types = disaster["incident_type"].value_counts().head(10).index

filtered = disaster[
    disaster["incident_type"].isin(top_types)
]

sns.boxplot(
    x="incident_type",
    y="year",
    data=filtered
)

plt.xticks(rotation=45)
plt.title("Incident Type vs Declaration Year")
plt.savefig(os.path.join(OUTPUT_DIR, "incident_type_vs_year.png"))
plt.clf()

print("Running Hypothesis Testing...\n")

air_quality["month"] = (
    np.arange(len(air_quality)) % 12
) + 1

monthly_avg_severity = (
    air_quality
    .groupby("month")["Severity_Score"]
    .mean()
    .reset_index()
)

monthly_avg_severity.rename(
    columns={"Severity_Score": "avg_severity_score"},
    inplace=True
)

monthly_disaster_counts = (
    disaster
    .groupby("month")
    .size()
    .reset_index(name="num_disasters")
)

merged_monthly_data = pd.merge(
    monthly_avg_severity,
    monthly_disaster_counts,
    on="month",
    how="outer"
)

merged_monthly_data.fillna(0, inplace=True)

merged_monthly_data.to_csv(
    os.path.join(OUTPUT_DIR, "merged_monthly_data.csv"),
    index=False
)

print("Merged Monthly Data:")
print(merged_monthly_data)

correlation_coefficient, p_value = stats.pearsonr(
    merged_monthly_data["avg_severity_score"],
    merged_monthly_data["num_disasters"]
)

print(
    f"\nPearson Correlation Coefficient: {correlation_coefficient:.4f}"
)

print(f"P-value: {p_value:.4f}")

alpha = 0.05

if p_value < alpha:
    print(
        "Reject Null Hypothesis (H0): "
        "There is a statistically significant relationship."
    )
else:
    print(
        "Fail to Reject Null Hypothesis (H0): "
        "No statistically significant relationship found."
    )

print("\nRunning Model Building...\n")

X = merged_monthly_data[["num_disasters"]]
y = merged_monthly_data["avg_severity_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)

print(f"R-squared Score: {r2:.4f}")

print("\nProject completed successfully.")
print("All visualizations saved in the output folder.")
print("Docker-ready analysis finished.")
