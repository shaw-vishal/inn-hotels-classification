# INN Hotels — Booking Cancellation Predictor

An end-to-end classification project predicting hotel booking cancellations, enabling INN Hotels to implement risk-based policies and reduce revenue volatility.

---

## Problem Statement

INN Hotels Group was experiencing a high and growing number of booking cancellations — directly impacting revenue stability and operational planning. With online booking platforms making cancellations frictionless for customers, the hotel needed a **data-driven way to identify high-risk bookings before they cancel**, so they could act proactively.

---

## Dataset

- **Records:** 36,275 bookings
- **Target variable:** `booking_status` — Cancelled (1) or Not Cancelled (0)
- **Class distribution:** 67.2% Cancelled · 32.8% Not Cancelled
- **Features include:** lead time, arrival month, market segment, meal plan, room type, avg price per room, number of special requests, repeated guest flag, previous cancellation history, number of adults/children, weekend/weekday nights

---

## Approach

**EDA:**
- Overall cancellation rate analysis
- Monthly booking distribution — seasonality patterns
- Market segment concentration analysis
- Lead time vs cancellation relationship
- Repeated guest vs first-time guest cancellation rates
- Special requests vs cancellation behavior
- Correlation matrix

**Preprocessing:**
- Removed `Booking_ID` (non-predictive)
- Label encoding for categorical variables
- 70/30 stratified train/test split
- Standard scaling for Logistic Regression
- VIF analysis — all values below 2, no multicollinearity

**Models built:**
1. Logistic Regression (baseline → threshold-tuned)
2. Decision Tree (baseline → cost-complexity pruned)

---

## Model Comparison

| Model | ROC AUC |
|-------|---------|
| Tuned Logistic Regression | **0.8607** ✅ |
| Decision Tree (Base) | 0.8602 |
| Decision Tree (Pruned) | 0.8560 |

**Final model: Tuned Logistic Regression** — selected for highest AUC and superior interpretability. Coefficients allow direct quantification of each feature's impact on cancellation probability.

---

## Final Model Performance

| Metric | Value |
|--------|-------|
| ROC AUC | **0.8607** |
| Accuracy | **79%** |
| Precision (Cancelled) | 0.88 |
| Recall (Cancelled) | 0.79 |
| F1-score (Cancelled) | 0.83 |

---

## Key Findings

| Factor | Impact on Cancellation |
|--------|----------------------|
| **Lead time** | Higher lead time → much higher cancellation risk |
| **Repeated guest** | First-time guests cancel at 66.4% vs repeated guests at only 1.7% |
| **Special requests** | 0 requests → 56.8% cancellation · 2 requests → 14.6% · 3+ → 0% |
| **Previous cancellations** | Strong positive predictor of future cancellation |
| **Market segment** | Segment 4 (64% of bookings) drives concentration risk |
| **Avg room price** | Higher price → marginally higher cancellation sensitivity |

---

## Business Recommendations

1. **High lead time bookings** (especially 150+ days) → require partial non-refundable deposit
2. **Customers with prior cancellation history** → stricter refund policy or higher deposit requirement
3. **Repeated guests** → maintain flexible terms, they cancel at <2% — protecting this segment builds loyalty
4. **Zero special requests** → trigger engagement prompts during booking to increase commitment
5. **Peak months (Aug–Oct)** → implement controlled overbooking within predicted cancellation range
6. **Segment 4 concentration** → diversify acquisition channels to reduce systemic risk

---

## Tech Stack

`Python` · `Scikit-learn` · `Statsmodels` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## Author

**Vishal Shaw** · Data Scientist  
[Portfolio](https://shaw-vishal.github.io) · [LinkedIn](https://linkedin.com/in/your-linkedin) · [Email](mailto:vishshaw6@gmail.com)
