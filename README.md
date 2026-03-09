#  Stock Price Prediction using Machine Learning

This project builds a **machine learning model to predict stock price movement** using historical stock market data. The objective is to determine whether the **next day's stock price will increase or decrease**, generating a **Buy / No Buy signal**. The model is trained using historical stock data from **Tesla Inc.** and applies multiple machine learning algorithms to learn market patterns.

---

##  Project Overview
Stock price prediction is a challenging task because markets are influenced by many factors such as trends, momentum, and investor behavior. This project demonstrates a **complete machine learning workflow**, including data analysis, feature engineering, technical indicators, model training, and prediction.

Main steps in the project:
1. Data loading and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature engineering and technical indicators  
4. Model training using multiple ML algorithms  
5. Model evaluation and comparison  
6. Prediction using user input  

---

##  Dataset
The dataset contains historical stock price data of **Tesla Inc.**

**Dataset features:**
- Date  
- Open  
- High  
- Low  
- Close  
- Volume  

Additional features are created from these values to improve prediction performance.

---

##  Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  

---

##  Machine Learning Models
The following models were trained and compared:

1. **Logistic Regression**  
2. **Support Vector Machine (SVM)**  
3. **XGBoost Classifier**

Among these models, **XGBoost achieved the best performance**.

---

##  Feature Engineering
To improve model accuracy, additional features were created.

### Price-Based Features
- `open-close`
- `low-high`

### Date-Based Features
- Day
- Month
- Year
- Quarter End Indicator

### Technical Indicators
**Moving Average (MA10)** – Average closing price of the last 10 trading days.  
**Moving Average (MA50)** – Average closing price of the last 50 trading days.  
**Relative Strength Index (RSI)** – Momentum indicator showing overbought or oversold conditions.  
**MACD (Moving Average Convergence Divergence)** – Measures trend changes using exponential moving averages.

---

##  Model Evaluation
Models were evaluated using the **ROC-AUC score**.

| Model | ROC-AUC Score |
|------|---------------|
| Logistic Regression | ~0.60 |
| SVM | ~0.58 |
| XGBoost | **~0.65 – 0.72** |

XGBoost produced the **best prediction performance**.

---

##  Data Visualization
Exploratory Data Analysis was performed using:
- Stock closing price trend plots
- Distribution plots
- Box plots for outlier detection
- Correlation heatmap

These visualizations help understand **stock behavior and feature relationships**.

---

##  User Input Prediction
The project allows users to manually enter stock parameters to generate predictions.

**Example Inputs**
- Open Price  
- Close Price  
- High Price  
- Low Price  
- Month  
- MA10  
- MA50  
- RSI  
- MACD  

**Output**
- `BUY SIGNAL` → Stock price expected to increase  
- `NO BUY SIGNAL` → Stock price expected to decrease  

---

## 📂 Project Structure
