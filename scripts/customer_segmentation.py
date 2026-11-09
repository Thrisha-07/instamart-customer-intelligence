
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. PATHS AND DATA
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(DATA_DIR / "clean" / "customers.csv")
orders = pd.read_csv(DATA_DIR / "clean" / "orders.csv")
products = pd.read_csv(DATA_DIR / "clean" / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["status"] = orders["status"].astype(str).str.lower()
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

completed = orders[orders["status"] == "completed"].copy()
completed["order_amount"] = completed["order_amount"].fillna(0)


# ---------------------------------------------------------
# 2. RFM TABLE
# ---------------------------------------------------------

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

rfm = rfm.dropna(subset=["recency", "frequency", "monetary"])
rfm[["recency", "frequency", "monetary"]] = rfm[
    ["recency", "frequency", "monetary"]
].astype(float)

print("\n========== RFM CHECKPOINTS ==========")
print("Customers:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe())
print("Recency median:", rfm["recency"].median())
print("Recency min/max:", rfm["recency"].min(), rfm["recency"].max())
print("Frequency mean/max:", rfm["frequency"].mean(), rfm["frequency"].max())
print("Monetary mean/max:", rfm["monetary"].mean(), rfm["monetary"].max())


# ---------------------------------------------------------
# 3. K-MEANS: TEST K = 2 TO 8
# ---------------------------------------------------------

rfm_features = ["recency", "frequency", "monetary"]

rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(rfm[rfm_features])

kmeans_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(rfm_scaled)

    kmeans_results.append({
        "K": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(rfm_scaled, labels),
    })

kmeans_table = pd.DataFrame(kmeans_results)
kmeans_table.to_csv(OUTPUT_DIR / "kmeans_metrics.csv", index=False)

print("\n========== K-MEANS COMPARISON ==========")
print(kmeans_table.to_string(index=False))

# K=4 is selected using the elbow, strongest reference silhouette,
# and the business usefulness of four customer groups.
FINAL_K = 4

final_kmeans = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42,
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)


# ---------------------------------------------------------
# 4. PROFILE AND NAME SEGMENTS
# ---------------------------------------------------------

profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

premium_segment = int(
    profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"]
)

# Names are assigned using observed segment characteristics.
# Cluster IDs themselves have no inherent ranking.
# Assign business names based on observed cluster behaviour.
for idx, row in profiles.iterrows():
    if int(row["segment_id"]) == premium_segment:
        name = "Premium High Spenders"

    elif row["mean_frequency"] >= 15:
        name = "Loyal Frequent Customers"

    elif row["mean_recency"] >= 60:
        name = "Dormant At-Risk Customers"

    else:
        name = "Occasional Value Customers"

    profiles.loc[idx, "segment_name"] = name

name_map = dict(zip(profiles["segment_id"], profiles["segment_name"]))

rfm["segment_name"] = rfm["segment_id"].map(name_map)
rfm["is_premium"] = rfm["segment_id"].eq(premium_segment)
segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].copy()

segment_output.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False,
)

print("\n========== SEGMENT PROFILES ==========")
print(profiles.to_string(index=False))
print("Premium segment ID:", premium_segment)
print("Premium segment name:", name_map[premium_segment])


# ---------------------------------------------------------
# 5. PCA ON SCALED RFM
# ---------------------------------------------------------

pca = PCA(n_components=2)
pca.fit(rfm_scaled)

print("\n========== PCA ==========")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())


# ---------------------------------------------------------
# 6. BUILD LAPSE COHORT AND LABELS
# ---------------------------------------------------------

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

pre_cutoff = orders[
    (orders["order_ts"] < CUTOFF)
    & (orders["status"] == "completed")
].copy()

cohort_ids = pre_cutoff["customer_id"].drop_duplicates()

# Label: 1 means no completed order during September 2025.
september_completed = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
]

september_customer_ids = set(september_completed["customer_id"])

labels = pd.DataFrame({"customer_id": cohort_ids})
labels["lapsed"] = (
    ~labels["customer_id"].isin(september_customer_ids)
).astype(int)

print("\n========== LAPSE COHORT ==========")
print("Cohort customers:", len(labels))
print("Lapsed customers:", int(labels["lapsed"].sum()))
print("Lapsed percentage:", round(labels["lapsed"].mean() * 100, 2))


# ---------------------------------------------------------
# 7. FEATURE ENGINEERING USING PRE-CUTOFF DATA ONLY
# ---------------------------------------------------------

pre_cutoff_all = orders[
    (orders["order_ts"] < CUTOFF)
    & (orders["status"].isin(["completed", "cancelled"]))
].copy()

completed_pre = pre_cutoff_all[
    pre_cutoff_all["status"] == "completed"
].copy()

cancelled_pre = pre_cutoff_all[
    pre_cutoff_all["status"] == "cancelled"
].copy()

completed_pre["order_amount"] = completed_pre["order_amount"].fillna(0)

completed_features = (
    completed_pre.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)

completed_features["recency"] = (
    CUTOFF - completed_features["last_order_ts"]
).dt.days

all_final_counts = (
    pre_cutoff_all.groupby("customer_id")
    .agg(total_final_orders=("order_id", "nunique"))
    .reset_index()
)

cancelled_counts = (
    cancelled_pre.groupby("customer_id")
    .agg(cancelled_orders=("order_id", "nunique"))
    .reset_index()
)

features = completed_features.merge(
    all_final_counts, on="customer_id", how="left"
).merge(
    cancelled_counts, on="customer_id", how="left"
)

features["cancelled_orders"] = features["cancelled_orders"].fillna(0)

features["cancellation_rate"] = (
    features["cancelled_orders"] / features["total_final_orders"]
)

# Find the most frequently purchased product category per customer.
category_orders = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)

def most_frequent_category(series):
    modes = series.dropna().mode()
    if len(modes) == 0:
        return "Unknown"
    return sorted(modes.astype(str))[0]

favourite_category = (
    category_orders.groupby("customer_id")["category"]
    .agg(most_frequent_category)
    .reset_index(name="favourite_category")
)

features = features.merge(
    favourite_category, on="customer_id", how="left"
)

customer_features = customers[
    ["customer_id", "city", "signup_date", "age", "gender"]
].copy()

customer_features["tenure_days"] = (
    CUTOFF - customer_features["signup_date"]
).dt.days

features = features.merge(
    customer_features[
        ["customer_id", "city", "age", "gender", "tenure_days"]
    ],
    on="customer_id",
    how="left",
)

features = features.merge(
    labels[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner",
)

# Explicit missing-value handling. The model pipelines also contain
# imputers fitted on training data only.
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

categorical_features = [
    "favourite_category",
    "city",
    "gender",
]

features[numeric_features] = features[numeric_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

for col in categorical_features:
    features[col] = features[col].fillna("Unknown").astype(str)

X = features[numeric_features + categorical_features].copy()
y = features["lapsed"].astype(int).copy()

assert not X.isnull().any().any(), "Missing feature values remain."
assert len(X) == len(y)

print("\nFeature missing values:")
print(X.isnull().sum())
print("All feature columns contain no missing values:", not X.isnull().any().any())


# ---------------------------------------------------------
# 8. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X,
    y,
    features["customer_id"],
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().to_dict())
print("Test class counts:")
print(y_test.value_counts().to_dict())


# ---------------------------------------------------------
# 9. PREPROCESSING PIPELINE
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
        )),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# ---------------------------------------------------------
# 10. BASELINE
# ---------------------------------------------------------

baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n========== BASELINE ==========")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print("Lapsed precision:", precision_score(
    y_test, baseline_predictions, zero_division=0
))
print("Lapsed recall:", recall_score(
    y_test, baseline_predictions, zero_division=0
))
print("Lapsed F1:", f1_score(
    y_test, baseline_predictions, zero_division=0
))


# ---------------------------------------------------------
# 11. LOGISTIC REGRESSION GRID SEARCH
# ---------------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

logistic_search.fit(X_train, y_train)
best_logistic = logistic_search.best_estimator_

print("\n========== LOGISTIC REGRESSION ==========")
print("Best parameters:", logistic_search.best_params_)
print("Best cross-validation F1:", logistic_search.best_score_)


# ---------------------------------------------------------
# 12. RANDOM FOREST GRID SEARCH
# ---------------------------------------------------------

forest_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )),
    ]
)

forest_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}

forest_search = GridSearchCV(
    forest_pipeline,
    forest_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

forest_search.fit(X_train, y_train)
best_forest = forest_search.best_estimator_

print("\n========== RANDOM FOREST ==========")
print("Best parameters:", forest_search.best_params_)
print("Best cross-validation F1:", forest_search.best_score_)


# ---------------------------------------------------------
# 13. TEST-SET EVALUATION
# ---------------------------------------------------------

def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_eval, predictions))
    print("Lapsed precision:", precision_score(
        y_eval, predictions, zero_division=0
    ))
    print("Lapsed recall:", recall_score(
        y_eval, predictions, zero_division=0
    ))
    print("Lapsed F1:", f1_score(
        y_eval, predictions, zero_division=0
    ))

    return predictions


print("\n========== UNTOUCHED TEST SET ==========")

logistic_predictions = evaluate_model(
    "Logistic Regression",
    best_logistic,
    X_test,
    y_test,
)

forest_predictions = evaluate_model(
    "Random Forest",
    best_forest,
    X_test,
    y_test,
)

logistic_test_f1 = f1_score(
    y_test, logistic_predictions, zero_division=0
)
forest_test_f1 = f1_score(
    y_test, forest_predictions, zero_division=0
)

# Select the model using cross-validation F1, not test-set performance.
if forest_search.best_score_ >= logistic_search.best_score_:
    selected_model = best_forest
    selected_model_name = "Random Forest"
else:
    selected_model = best_logistic
    selected_model_name = "Logistic Regression"

# Evaluate the selected model on the untouched test set.
selected_test_f1 = f1_score(
    y_test,
    selected_model.predict(X_test),
    zero_division=0
)

print("\nSelected model:", selected_model_name)
print(
    "Selected cross-validation F1:",
    max(logistic_search.best_score_, forest_search.best_score_)
)
print("Selected test F1:", selected_test_f1)

if selected_test_f1 < 0.55:
    print("WARNING: Selected model is below the assignment F1 target of 0.55.")

# ---------------------------------------------------------
# 14. CLASS-BALANCE EXPERIMENT
# ---------------------------------------------------------

training_data = X_train.copy()
training_data["lapsed"] = y_train.values

majority_count = int(y_train.value_counts().max())
minority_count = int(y_train.value_counts().min())

minority_rows = training_data[
    training_data["lapsed"] == 1
]

extra_lapsed = minority_rows.sample(
    n=majority_count - minority_count,
    replace=True,
    random_state=42,
)

balanced_training = pd.concat(
    [training_data, extra_lapsed],
    ignore_index=True,
)

X_balanced = balanced_training.drop(columns=["lapsed"])
y_balanced = balanced_training["lapsed"]

balanced_logistic = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            C=logistic_search.best_params_["model__C"],
        )),
    ]
)

balanced_logistic.fit(X_balanced, y_balanced)

balanced_predictions = balanced_logistic.predict(X_test)

print("\n========== CLASS BALANCE EXPERIMENT ==========")
print("Original training class counts:", y_train.value_counts().to_dict())
print("Balanced training class counts:", y_balanced.value_counts().to_dict())
print("Original logistic precision:", precision_score(
    y_test, logistic_predictions, zero_division=0
))
print("Original logistic recall:", recall_score(
    y_test, logistic_predictions, zero_division=0
))
print("Original logistic F1:", f1_score(
    y_test, logistic_predictions, zero_division=0
))
print("Balanced logistic precision:", precision_score(
    y_test, balanced_predictions, zero_division=0
))
print("Balanced logistic recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))
print("Balanced logistic F1:", f1_score(
    y_test, balanced_predictions, zero_division=0
))


# ---------------------------------------------------------
# 15. FEATURE IMPORTANCE AND LOGISTIC COEFFICIENTS
# ---------------------------------------------------------

forest_preprocessor = best_forest.named_steps["preprocess"]
forest_model = best_forest.named_steps["model"]

forest_feature_names = forest_preprocessor.get_feature_names_out()

importance_table = pd.DataFrame({
    "feature": forest_feature_names,
    "importance": forest_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n========== RANDOM FOREST TOP FIVE FEATURES ==========")
print(importance_table.head(5).to_string(index=False))

logistic_preprocessor = best_logistic.named_steps["preprocess"]
logistic_model = best_logistic.named_steps["model"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_table = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_model.coef_[0],
})

print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

for _, row in coefficient_table.reindex(
    coefficient_table["coefficient"].abs().sort_values(
        ascending=False
    ).index
).head(3).iterrows():
    print(row["feature"], row["coefficient"])


# ---------------------------------------------------------
# 16. SAVE, RELOAD AND VERIFY MODEL
# ---------------------------------------------------------

model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions,
)

print("\n========== MODEL PERSISTENCE ==========")
print("Saved model:", model_path)
print("Reloaded predictions exactly match:", predictions_match)

assert predictions_match, "Reloaded model predictions do not match."


# ---------------------------------------------------------
# 17. PREMIUM CUSTOMER RETENTION LIST
# ---------------------------------------------------------

test_results = pd.DataFrame({
    "customer_id": test_ids.values,
    "lapsed": y_test.values,
})

test_results["is_premium"] = test_results["customer_id"].isin(
    segment_output.loc[
        segment_output["is_premium"], "customer_id"
    ]
)

test_results["lapse_probability"] = selected_model.predict_proba(
    X_test
)[:, 1]

premium_test = test_results[
    test_results["is_premium"]
].copy()

retention_list = premium_test.sort_values(
    "lapse_probability",
    ascending=False,
).head(10)

retention_list = retention_list.rename(
    columns={"lapsed": "actually_lapsed"}
)

retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention_list.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False,
)

print("\n========== RETENTION LIST ==========")
print(retention_list.to_string(index=False))
print(
    "True lapses in retention list:",
    int(retention_list["actually_lapsed"].sum()),
)
print(
    "Premium test customers:",
    len(premium_test),
)

print(
    "\nNOTE: The Premium flag is based on whole-window RFM, "
    "including September 2025. The retention list is illustrative "
    "and is not a leakage-free evaluation."
)

print("\nPart B processing complete.")
