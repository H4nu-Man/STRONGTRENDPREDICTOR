import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/content/Tesla.csv")

df.head()

df.shape

df.info()

df.describe()

plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title("Tesla Closing Price Trend")
plt.ylabel("Price")
plt.show()

(df['Close'] == df['Adj Close']).sum()

df.drop('Adj Close', axis=1, inplace=True)

df.isnull().sum()

features = ['Open','High','Low','Close','Volume']

plt.figure(figsize=(20,10))

for i,col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.histplot(df[col], kde=True)

plt.show()

plt.figure(figsize=(20,10))

for i,col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=df[col])

plt.show()

df['Date'] = pd.to_datetime(df['Date'])

df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)

data_grouped = df.drop('Date',axis=1).groupby('year').mean()

plt.figure(figsize=(20,10))

for i,col in enumerate(['Open','High','Low','Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()

plt.show()

df.drop('Date',axis=1).groupby('is_quarter_end').mean()

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

df['target'] = np.where(df['Close'].shift(-1) > df['Close'],1,0)

plt.pie(df['target'].value_counts(),
        labels=[0,1],
        autopct='%1.1f%%')

plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(10,8))

sns.heatmap(df.drop('Date',axis=1).corr(),
            annot=True,
            cmap='coolwarm')

plt.show()

df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()

delta = df['Close'].diff()

gain = (delta.where(delta>0,0)).rolling(14).mean()
loss = (-delta.where(delta<0,0)).rolling(14).mean()

RS = gain/loss

df['RSI'] = 100 - (100/(1+RS))

EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
EMA26 = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = EMA12 - EMA26

df = df.dropna()

features = df[['open-close',
               'low-high',
               'is_quarter_end',
               'MA10',
               'MA50',
               'RSI',
               'MACD']]

target = df['target']

scaler = StandardScaler()

features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features,
    target,
    test_size=0.1,
    random_state=2022
)

print(X_train.shape, X_valid.shape)

models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier()
]

for model in models:

    model.fit(X_train, Y_train)

    train_pred = model.predict_proba(X_train)[:,1]
    valid_pred = model.predict_proba(X_valid)[:,1]

    print(model)
    print("Training ROC-AUC:", roc_auc_score(Y_train, train_pred))
    print("Validation ROC-AUC:", roc_auc_score(Y_valid, valid_pred))
    print()

ConfusionMatrixDisplay.from_estimator(
    models[0],
    X_valid,
    Y_valid
)

plt.show()

print("Enter Stock Data")

open_price = float(input("Open Price: "))
close_price = float(input("Close Price: "))
high_price = float(input("High Price: "))
low_price = float(input("Low Price: "))
month = int(input("Month (1-12): "))

open_close = open_price - close_price
low_high = low_price - high_price

is_quarter_end = 1 if month%3==0 else 0

MA10 = float(input("MA10 value: "))
MA50 = float(input("MA50 value: "))
RSI = float(input("RSI value: "))
MACD = float(input("MACD value: "))

user_data = np.array([[open_close, low_high, is_quarter_end,
                       MA10, MA50, RSI, MACD]])

user_data = scaler.transform(user_data)

prediction = models[0].predict(user_data)

if prediction[0]==1:
    print("BUY SIGNAL")
else:
    print("NO BUY SIGNAL")
