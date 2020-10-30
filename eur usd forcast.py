# import quandl, pandas, numpy and all other module need to done my job.
#We'll be using the numpy module to convert data to numpy arrays,
#which is what Scikit-learn wants. We will talk more on preprocessing and cross_validation 
#when we get to them in the code, but preprocessing is the module used to do some cleaning/scaling of data prior to machine learning, and 
#cross_ alidation is used in the testing stages.
#Finally, we're also importing the LinearRegression algorithm as well as svm from Scikit-learn,
#which we'll be using as our machine learning algorithms to demonstrate results.
import pandas as pd
import numpy as np
import quandl
import math
from sklearn import metrics 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

style.use('ggplot')

#3df = quandl.get("GDAX/ETH_EUR", authtoken="D4W6ZcCw2gAf4bzSLZ2j")
#df = pd.read_csv('eurusd.csv')
df = pd.read_csv("G:\code\eurusd.csv", parse_dates=True)

print(df.head())
print(df.tail())

df.columns =['date','open','high','low','close','volume']

df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]

print(df.head())

#Here, we define the forecasting column, then we fill any NaN data with -99999. 
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)


X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

#it is included in the preprocessing module of Scikit-Learn. To utilize this applying preprocessing.scale to your X variable:
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


from sklearn.ensemble import RandomForestRegressor

#Scikit-Learn (sklearn),  train with .fit
clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)  

print('accuracy ',accuracy)
#The forecast_set is an array of forecasts, showing that not only could you just seek out a single prediction  
forecast_set = clf.predict(X_lately)
print('forecast_set\n',forecast_set)


df['Forecast'] = np.nan

#recall that we predict 10% out into the future, and we saved that last 10% of our data to do this, 
#thus, we can begin immediately predicting since -10% has data that we can predict 10% out and be the next prediction
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 3600
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 3600
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#print(df)
print(df.head())
print(df.tail())


# Nicher ei 2 lines kaj kore na
#print(classification_report(y_test, y_pred))
#print('confusion_matrix:',confusion_matrix(y_test, y_pred))


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

df['open'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()    
    



