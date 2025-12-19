# Comparative Volatility and Predictive Performance of UAE and U.S. Stock Markets (2022–2023)

*Analyzing market behavior, volatility, and predictive modeling across two global financial markets.*

---

## Introduction

This project studies and compares stock market behavior in the **United Arab Emirates (ADX)** and the **United States (NYSE)** during **2022–2023**.  
The analysis has two main goals:

1. **Market Analysis:**  
   Compare volatility, returns, sector behavior, and correlations across banking and energy stocks in both markets.

2. **Predictive Modeling:**  
   Build and evaluate machine learning models that predict **next-day stock price direction (Up vs Down)** using historical returns and rolling features.

We combine **exploratory data analysis (EDA)** with **classification models** to understand both market structure and predictability.  
Our results show that the **U.S. market is more predictable**, while the **UAE market appears noisier**, especially for short-term indicators.

---

## Project Question

**How do the U.S. and UAE stock markets differ in volatility, returns, and predictability during 2022–2023, and which machine learning models perform best in forecasting next-day price direction?**

---

## Data Description

### Data Sources

Daily stock price data was collected for four companies representing **Banking** and **Energy** sectors in both markets:

| Market | Sector | Company | Ticker | Source |
|------|------|------|------|------|
| UAE (ADX) | Banking | First Abu Dhabi Bank | FAB | ADX |
| UAE (ADX) | Energy | Abu Dhabi National Energy Co. | TAQA | ADX |
| U.S. (NYSE) | Banking | JPMorgan Chase | JPM | Yahoo Finance |
| U.S. (NYSE) | Energy | ExxonMobil | XOM | Yahoo Finance |

The dataset covers **January 2022 – December 2023**, with slightly different numbers of trading days due to market holidays.

### Feature Engineering

We created several features to support prediction:

- **Daily Returns**
- **Lagged Returns** (1-day, 2-day, 3-day)
- **Rolling Mean Returns** (5-day)
- **Rolling Volatility** (5-day standard deviation)
- **Categorical Variables** (Market, Sector, Company → one-hot encoded)
- **Binary Target Variable**:
  - `UpDay = 1` if next-day return > 0  
  - `UpDay = 0` otherwise

After cleaning missing values, the final dataset contained **1,986 observations and 10 predictive features**.

---

## Models and Methods

We framed the task as a **binary classification problem** and split the data using a **time-based split**:
- **Training:** Before July 1, 2023  
- **Testing:** July 1, 2023 onward

All models were evaluated on **out-of-sample test data**.

### Why Accuracy Is Not Enough

Because the number of Up and Down days is **not perfectly balanced**, accuracy alone can be misleading.  
For this reason, we also computed:

- **Precision:** How often predicted Up days were actually Up  
- **Recall:** How many true Up days were correctly identified  
- **F1-Score:** A balance between precision and recall  

These metrics give a clearer picture of real predictive performance.

---

## Models Evaluated

### Logistic Regression

- Serves as a **baseline linear model**
- Works well when relationships are simple and stable
- Performance:
  - **Accuracy:** ~0.74  
  - **Precision:** ~0.79  
  - **Recall:** ~0.66  

**Interpretation:**  
Logistic Regression performed surprisingly well, suggesting that lagged returns and rolling features contain meaningful linear signals, especially in the U.S. market.

---

### K-Nearest Neighbors (KNN)

- Uses similarity between observations
- Sensitive to noise and feature scaling
- Performance:
  - **Accuracy:** ~0.63  
  - **Precision:** ~0.64  
  - **Recall:** ~0.58  

**Why performance was weaker:**  
Stock market data is noisy and high-variance. KNN struggles because nearby points are not consistently predictive in financial time series.

---

### Decision Tree

- Captures non-linear patterns
- Easy to interpret but prone to overfitting
- Performance:
  - **Accuracy:** ~0.67  
  - **Precision:** ~0.71  
  - **Recall:** ~0.58  

**Interpretation:**  
Decision Trees improved over KNN by capturing interactions between features, but limited depth was necessary to prevent overfitting.

---

### Random Forest

- Ensemble of multiple decision trees
- Reduces overfitting and improves generalization
- Performance:
  - **Accuracy:** ~0.66  
  - **Precision:** ~0.72  
  - **Recall:** ~0.53  

**Market-Specific Results:**

| Market | Accuracy | Precision | Recall | F1-Score |
|------|------|------|------|------|
| UAE (ADX) | ~0.63 | ~0.64 | ~0.43 | ~0.52 |
| U.S. (NYSE) | ~0.69 | ~0.79 | ~0.60 | ~0.68 |

**Key Insight:**  
The Random Forest model performed significantly better for U.S. stocks, indicating higher short-term predictability compared to the UAE market.

---

### Gradient Boosting

- Sequentially corrects previous model errors
- Strong performance on structured tabular data
- Performance:
  - **Accuracy:** ~0.70  
  - **Precision:** ~0.72  
  - **Recall:** ~0.68  

**Interpretation:**  
Gradient Boosting achieved the best balance between accuracy and recall, showing strong ability to detect Up days while controlling false positives.

---

### Neural Network

- Multi-layer feedforward network
- Requires large datasets and careful tuning
- Performance:
  - **Accuracy:** ~0.62  

**Why performance was lower:**  
The dataset size is relatively small for neural networks, and financial data is noisy. As a result, simpler models generalized better.

---

## Results and Interpretation

### Overall Performance Comparison

- **Best Overall Model:** Gradient Boosting  
- **Strong Baseline:** Logistic Regression  
- **Weaker Models:** KNN and Neural Network  

### Market Differences

- The **U.S. market** showed:
  - Higher accuracy
  - Higher recall
  - Better F1-scores
- The **UAE market**:
  - Appeared noisier
  - Was harder to predict using short-term indicators
  - Was strongly influenced by energy sector volatility

### Key Insight from Metrics

F1-scores were consistently higher for U.S. stocks, showing that **lagged returns and rolling volatility contain more predictive power in the U.S. market** than in the UAE.

---

## Conclusion

This project demonstrates that **market structure matters** for predictability.  
While the UAE market is often viewed as stable, its strong dependence on the energy sector introduces volatility that reduces short-term forecasting accuracy.

By combining **EDA, multiple models, and evaluation beyond accuracy**, we showed that:

- Simple models can outperform complex ones in noisy financial data
- Precision and recall are essential when predicting directional movement
- The U.S. market offers clearer short-term signals than the UAE market

The comparison of models and markets highlights how **sector dominance, volatility, and market maturity** influence predictive success.

---

## Next Steps

- Add more sectors (telecom, real estate, consumer goods)
- Expand to other global markets (Europe, Asia)
- Include macroeconomic indicators (interest rates, oil prices)
- Test longer rolling windows and alternative targets
- Explore probability-based trading strategies

---

## References

- Abu Dhabi Securities Exchange (ADX)
- Yahoo Finance API
- Scikit-learn Documentation
- IMF World Economic Outlook (2023)
- Bloomberg Market Reports
