# Project Status — ML Apartment Rents Zurich

Last updated: 2026-05-08

---

## Pipeline Status

Notebook | Status | Output
01 Data Ingestion | Done | listings_raw.csv — 6,718 rows (2 snapshots: Jun + Sep 2025)
02 EDA | Done | listings_eda.csv — 5,095 rows after price filter
03 Preprocessing | Done | X_train / X_test — 89 features, outlier cap at 800 CHF
04 Modeling | Done | rent_pipeline.pkl — Random Forest retrained
05 Evaluation + XAI | Done | MAE / RMSE / R² + SHAP plots

---

## Model Performance

Model | MAE | RMSE | R²
Linear Regression (baseline, v1) | 77.40 CHF | 117.30 CHF | 0.254
Random Forest (v1 — 11 features, 1 snapshot) | 51.85 CHF | 100.89 CHF | 0.448
Random Forest (v2 — 22 features, 2 snapshots) | 37.04 CHF | 66.32 CHF | 0.718

What the metrics mean:

MAE (Mean Absolute Error) — average CHF the model is wrong per listing
  37.04 CHF means: on a typical 139 CHF/night listing the model is off by ~27%.
  MAE treats all errors equally — a 10 CHF miss counts the same as a 100 CHF miss.
  Lower is better. Use MAE to explain model accuracy in plain language.

RMSE (Root Mean Squared Error) — like MAE but punishes large errors much harder
  Because errors are squared before averaging, one prediction off by 200 CHF
  hurts far more than ten predictions off by 20 CHF.
  66.32 CHF RMSE vs 37.04 CHF MAE means: most predictions are close,
  but a few listings are harder to predict (e.g. unusual luxury properties).
  Lower is better. RMSE is more sensitive to outliers than MAE.

R² (R-squared / Coefficient of Determination) — how much price variance the model explains
  R² = 0.718 means the model explains 71.8% of why prices differ between listings.
  The remaining 28.2% is driven by factors we do not have in the data:
  photo quality, host personality, review text, seasonal demand spikes.
  Scale: 0.0 = no better than guessing the mean / 1.0 = perfect prediction.
  Anything above 0.6 is considered good for real-world pricing data.

v2 improvements over v1:
- MAE:  51.85 → 37.04 CHF  (-29%)
- RMSE: 100.89 → 66.32 CHF  (-34%)
- R²:    0.448 → 0.718       (+60%)

Key changes that drove the improvement:
- Outlier capping at 99th percentile (800 CHF) — most critical fix; a handful of
  9,000 CHF/night listings were destroying RMSE without contributing learnable signal
- 2 snapshots instead of 1 (Jun + Sep 2025) — 5,095 usable rows vs 2,592;
  Random Forest benefits directly from more training examples
- 22 features instead of 11 — amenities_count, property_type, host quality signals,
  number_of_reviews_ltm and snapshot_month give the model richer pricing signals
- 3 engineered features: amenities_count (from raw list), host_since_years (from date),
  snapshot_month (temporal context)

---

## Plots — What They Show

price_distribution.png
  Two histograms side by side.
  Left: full price range — shows extreme right skew, long tail to 10,000 CHF.
  Right: prices below 500 CHF — shows the realistic market; peak around 100-150 CHF.
  Takeaway: most Zurich Airbnb listings cluster between 80-200 CHF/night.
  The long tail justifies the outlier capping decision.

price_by_room_type.png
  Boxplot of price grouped by room type (entire home / private room / shared room).
  Shows entire home/apt is systematically more expensive than private rooms.
  Justifies including room_type as a feature.
  Boxes show IQR (middle 50% of prices), whiskers show the spread, dots are outliers.

correlation_matrix.png
  Heatmap of Pearson correlations between all numeric features and price.
  Values range from -1 (perfect negative) to +1 (perfect positive).
  Warm colours = positive correlation, cool = negative.
  Key finding: bedrooms and accommodates have strongest correlation with price (~0.4-0.5).
  Note: Pearson assumes linear relationships and normality — Spearman would be
  more appropriate for skewed features like minimum_nights and number_of_reviews.

residuals.png
  Scatter plot of predicted price (x-axis) vs prediction error (y-axis).
  Each dot = one listing in the test set.
  Red line at y=0 = perfect prediction.
  Ideal pattern: dots scattered randomly around y=0 at all price levels.
  What to watch for: funnel shape (heteroscedasticity — errors grow with price),
  or systematic curves (the model misses non-linear patterns).

shap_summary.png
  SHAP (SHapley Additive exPlanations) dot plot — explains which features
  push predictions up or down for each individual listing.
  X-axis: SHAP value = CHF impact on predicted price.
  Colour: red = high feature value, blue = low feature value.
  Each dot = one listing. Features ordered by total importance (top = most important).
  Example: a red dot for bedrooms at +30 means a listing with many bedrooms
  gets its price pushed up by 30 CHF by that feature alone.

shap_bar.png
  Same SHAP data as summary plot, but simplified to mean absolute impact per feature.
  One bar per feature, length = average CHF influence across all test listings.
  Easier to read than the dot plot — use this for presentations.
  Top features from v1: bedrooms, accommodates, latitude, room_type, neighbourhood.
  v2 will show whether new features (amenities_count, property_type) earned their place.

---

## Dataset

Inside Airbnb — Zurich, June 2025 + September 2025 snapshots
6,718 raw rows → 5,095 usable rows · 22 features · Target: price/night (CHF, capped at 800)

Features (22 total — updated from original 11):

PROPERTY SIZE AND CAPACITY

bedrooms
  Why: Strongest single price driver (SHAP #1 in v1). More bedrooms = larger listing = higher price.
  Directly measurable, few missing values. No engineering needed.

accommodates
  Why: Maximum number of guests the listing fits. Strong price driver (SHAP #2).
  Correlated with bedrooms but adds independent signal — a studio can sleep 4 with sofa beds.

beds
  Why: Complements bedrooms. A listing with 2 bedrooms but 4 beds signals higher capacity.
  ~24% missing values — imputed with median (1 bed).

bathrooms
  Why: Size and comfort proxy. More bathrooms = larger, higher-end listing.
  ~24% missing — imputed with median (1 bathroom).

amenities_count  [ENGINEERED from raw amenities list]
  Why: The raw amenities column is a JSON list of strings (Wifi, Kitchen, Pool, etc.).
  We count the total number of amenities per listing. More amenities = higher quality = higher price.
  Engineered by applying ast.literal_eval() and counting list length.
  No missing values (empty list = 0).

LOCATION

neighbourhood_cleansed
  Why: Official Zurich district (Rathaus, Langstrasse, Seefeld, etc.). SHAP #5 in v1.
  Prices differ systematically by neighbourhood — Seefeld is more expensive than Altstetten.
  Used over raw neighbourhood because it has zero missing values (2,109 missing in raw version).
  One-hot encoded into binary columns per district.

latitude
  Why: Continuous north-south position within Zurich. SHAP #3 in v1.
  Captures price gradients that neighbourhood alone misses (e.g. lakefront vs hillside
  within the same district). Works together with longitude for a full geo signal.

longitude
  Why: Continuous east-west position. Together with latitude gives the model
  a precise spatial coordinate without needing manual geo-feature engineering.

LISTING TYPE

room_type
  Why: Entire home vs private room vs shared room — the most categorical price split. SHAP #4.
  Private rooms are systematically cheaper than entire homes regardless of size or location.
  One-hot encoded: Hotel room / Private room / Shared room (Entire home = reference category).

property_type
  Why: Apartment vs house vs loft vs boat vs villa etc.
  Captures premium property types (villas, penthouses) that room_type misses.
  Has many categories (30+) — one-hot encoded, rare types naturally get near-zero weight.
  New in v2: was excluded before because not in cat_cols; adding it improved R².

HOST QUALITY

host_is_superhost
  Why: Airbnb's official quality badge for hosts with high ratings and response rates.
  Superhosts tend to price higher and attract more bookings — a trust premium.
  Originally stored as t/f string — converted to 1/0 binary.

host_acceptance_rate
  Why: How often the host accepts booking requests (e.g. 81%).
  Low acceptance rate = selective host = possibly higher-end or premium listing.
  Originally stored as percentage string (81%) — converted to float (0.81).
  Missing values imputed with median.

host_response_rate
  Why: How reliably the host responds to enquiries (e.g. 100%).
  Guests pay a small premium for responsive hosts — reduces booking uncertainty.
  Same format conversion as acceptance rate (% string → float).

host_since_years  [ENGINEERED from host_since date]
  Why: How many years ago the host joined Airbnb, relative to 2025-09-01.
  Experienced hosts understand the market better and tend to price more strategically.
  Engineered by parsing the host_since date string and computing elapsed years.
  Missing values (no join date) imputed with median (~6 years).

calculated_host_listings_count
  Why: How many Airbnb listings this host manages.
  Professional hosts (many listings) behave differently from private hosts (1 listing) —
  they price more dynamically and maintain quality more consistently.
  Used over raw host_listings_count because Airbnb calculates this more reliably.

BOOKING RULES AND AVAILABILITY

minimum_nights
  Why: Minimum booking length the host requires.
  Short-stay listings (1-3 nights) target tourists and price accordingly.
  Monthly-minimum listings (28+ nights) are effectively furnished rentals at lower nightly rates.
  This feature captures the target market of the listing.

instant_bookable
  Why: Whether guests can book without host approval.
  Instant-bookable listings are often priced slightly lower to compensate for reduced control,
  or attract higher volume at competitive prices.
  Originally t/f string — converted to 1/0 binary.

availability_365
  Why: How many days per year the listing is available for booking.
  Active listings (high availability) differ from dormant or semi-private ones.
  Proxy for how seriously the host treats Airbnb as a revenue source.

REVIEWS AND TRUST

review_scores_rating
  Why: Overall guest satisfaction score (1.0–5.0). Listings with higher ratings
  have proven quality and can command a price premium.
  ~24% missing (listings with no reviews yet) — imputed with median (4.8).
  New in v2: was excluded before due to missing values, now included with imputation.

number_of_reviews
  Why: Total review count — proxy for listing maturity and popularity.
  A listing with 200 reviews is proven and trusted; one with 0 is uncertain.
  Correlated with reviews_per_month but adds a different signal (total history vs rate).

number_of_reviews_ltm
  Why: Reviews in the last 12 months. More current than total count —
  captures whether a listing is actively used right now vs historically popular.
  New in v2: directly available in dataset, no engineering needed.

reviews_per_month
  Why: Activity rate — how many reviews the listing receives per month on average.
  Complements total reviews with a time dimension. A listing with 50 reviews in
  1 month is very different from one with 50 reviews over 10 years.
  ~24% missing (new listings) — imputed with median.

TEMPORAL

snapshot_month
  Why: Which data snapshot the listing comes from (2025-06 or 2025-09).
  June and September may show different pricing due to summer tourism vs early autumn.
  Engineered during ingestion by labelling each CSV with its snapshot date.
  One-hot encoded — 2025-09 gets its own binary column (2025-06 = reference).
  New in v2: only possible with multiple snapshots.

---

## Correlation Methods

The current implementation uses Pearson for the full correlation matrix.
This is methodologically incorrect for several features in the dataset.
The following outlines the correct approach per variable type.

### Variable Types and Issues with Pearson

| Feature | Type | Issue with Pearson |
|---|---|---|
| `bedrooms` | ordinal | intervals not equidistant |
| `accommodates` | ordinal | same issue |
| `room_type` | nominal categorical | not applicable |
| `neighbourhood_cleansed` | nominal categorical | not applicable |
| `latitude` / `longitude` | metric | Pearson ok |
| `price` | metric, right-skewed | normality violated |
| `minimum_nights` | metric, heavily skewed | normality violated |
| `number_of_reviews` | metric, right-skewed | normality violated |

### Spearman Rank Correlation
- **When:** ordinal vs ordinal, or metric but non-normal distribution
- **Why better here:** rank-based, robust against outliers and skew
- **Relevant pairs:** `bedrooms` vs `price`, `accommodates` vs `price`
- The log-transform on price was a valid workaround, but Spearman is
  the cleaner solution without requiring a transform

```python
from scipy.stats import spearmanr

# single pair
corr, pvalue = spearmanr(df["bedrooms"], df["price"])

# full matrix for ordinal and metric columns
spearman_cols = ["bedrooms", "accommodates", "bathrooms",
                 "minimum_nights", "availability_365",
                 "number_of_reviews", "reviews_per_month", "price_log"]
df[spearman_cols].corr(method="spearman")
```

### Kendall's Tau
- **When:** ordinal vs ordinal, many ties in values
- **Why relevant here:** `bedrooms` has many ties (many listings with 1 bedroom)
  — Kendall is more robust than Spearman in this case
- **Tradeoff:** computationally expensive on large datasets

```python
df[spearman_cols].corr(method="kendall")
```

### Cramér's V
- **When:** categorical vs categorical (nominal)
- **Relevant pairs:** `room_type` vs `neighbourhood_cleansed`
- Returns a value between 0 and 1, analogous to Pearson but for categories

```python
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))

cramers_v(df["room_type"], df["neighbourhood_cleansed"])
```

### Point-Biserial Correlation
- **When:** binary (0/1) vs metric
- **Relevant pairs:** `is_private_room` vs `price`
- Mathematically identical to Pearson on binary variables, but conceptually correct

```python
from scipy.stats import pointbiserialr

corr, pvalue = pointbiserialr(df["is_private_room"], df["price"])
```

### Mutual Information (model-agnostic)
- **When:** any variable type, captures non-linear relationships
- All methods above miss non-linear dependencies — Mutual Information does not
- Useful as an additional feature selection step before modeling

```python
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series)
```

### Recommended Strategy: Split Matrix by Variable Type

```python
# 1. Metric vs Metric — Pearson
pearson_cols = ["latitude", "longitude", "price_log"]
corr_pearson = df[pearson_cols].corr(method="pearson")

# 2. Ordinal / skewed metric — Spearman
spearman_cols = ["bedrooms", "accommodates", "bathrooms",
                 "minimum_nights", "availability_365",
                 "number_of_reviews", "reviews_per_month", "price_log"]
corr_spearman = df[spearman_cols].corr(method="spearman")

# 3. Categorical — Cramér's V (compute pairwise manually)
# use cramers_v() above for room_type vs neighbourhood_cleansed
```

---

## Possible Dataset Extensions

### 1. Inside Airbnb — Additional Snapshots
- **What:** Multiple historical Zurich snapshots instead of only September 2025
- **Where:** http://insideairbnb.com/get-the-data/ — Zurich has several snapshots per year
- **Advantage:** Same structure, immediately compatible, no new cleaning logic needed
- **Effort:** Minimal — stack multiple CSVs and add a timestamp column
- **Caveat:** Same source, same bias

```python
import glob
dfs = [pd.read_csv(f).assign(snapshot_date=f.split("_")[1])
       for f in glob.glob("data/listings_*.csv")]
df_combined = pd.concat(dfs, ignore_index=True)
```

### New Variable Types After Extension

| New Feature | Type | Correct Method |
|---|---|---|
| `source` (airbnb/homegate) | binary | Point-Biserial vs price |
| `rental_type` (short/long term) | nominal | Cramér's V |
| `floor` | ordinal | Spearman |
| `build_year` | metric | Pearson |
| `distance_to_hb` (geo join) | metric | Pearson |

Always add `source` as a feature and test its predictive power —
it reveals whether platforms systematically cover different price segments.

### Common Schema for Multi-Source Integration

```python
COMMON_SCHEMA = {
    "price_chf":     float,   # target variable, normalised
    "bedrooms":      int,
    "accommodates":  int,
    "latitude":      float,
    "longitude":     float,
    "room_type":     str,     # "entire_home", "private_room", "shared_room"
    "source":        str,     # "airbnb", "homegate", "booking"
    "snapshot_date": str,
}
```

---

## Q&A Talking Point (Kolloquium)
If asked about Pearson on ordinal data:

> "We used Pearson as an exploratory baseline, but we are aware that Spearman
> would be the methodologically correct choice for ordinal features like
> bedrooms and accommodates. For this dataset size and feature set the rank
> correlations produce similar results, but in an extended dataset with
> additional sources we would switch to a type-aware correlation strategy."