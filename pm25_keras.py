# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from keras import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, BatchNormalization, Activation, Dropout 

# get titanic & test csv files as a DataFrame
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
train_df = pd.read_csv(SCRIPT_PATH + "/train.csv")

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
test_df = pd.read_csv(SCRIPT_PATH + "/test.csv")
dd = test_df['day']
combine = [train_df, test_df]

#change day from year/month/day to sorted by season
train_df['year'] = train_df['day']
train_df['month'] = train_df['day']
train_df['date'] = train_df['day']
for i in range(train_df.shape[0]):
    year,month,day = (train_df['day'][i]).split("/")
    month = int(month)
    year = int(year)
    day = int(day)
    train_df.at[i,'year'] = year
    train_df.at[i,'month'] = month
    train_df.at[i,'date'] = day

test_df['year'] = test_df['day']
test_df['month'] = test_df['day']
test_df['date'] = test_df['day']     
for i in range(test_df.shape[0]):
    year,month,day = (test_df['day'][i]).split("/")
    month = int(month)
    year = int(year)
    day = int(day)
    test_df.at[i,'year'] = year
    test_df.at[i,'month'] = month
    test_df.at[i,'date'] = day

train_df = train_df.drop(['day'], axis=1)
test_df = test_df.drop(['day'], axis=1)
combine = [train_df, test_df]
scaler = MinMaxScaler(feature_range=(0,1))

#for dataset in combine:    
#    dataset.loc[(dataset['hour'] > 21) & (dataset['hour'] <= 6), 'hour'] = 1
#    dataset.loc[(dataset['hour'] > 6) & (dataset['hour'] <= 10), 'hour'] = 2
#    dataset.loc[(dataset['hour'] > 11) & (dataset['hour'] <= 17), 'hour'] = 3
#    dataset.loc[(dataset['hour'] > 17) & (dataset['hour'] <= 21), 'hour'] = 4
    
for dataset in combine:
    dataset['wind'] = 0
    dataset.loc[ dataset['wind_ne'] == 1, 'wind'] = 1
    dataset.loc[ dataset['wind_se'] == 1, 'wind'] = 1
    dataset.loc[ dataset['wind_cv'] == 1, 'wind'] = 0
    dataset.loc[ dataset['wind_nw'] == 1, 'wind'] = 1
        
train_df = train_df.drop(['wind_ne', 'wind_nw', 'wind_cv', 'wind_se', 'year'], axis=1)
test_df = test_df.drop(['wind_ne', 'wind_nw', 'wind_cv', 'wind_se', 'year'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("pm2.5", axis=1)
Y_train = train_df["pm2.5"]
X_train = MinMaxScaler().fit_transform(X_train)
X_test  = test_df.copy()
X_test = MinMaxScaler().fit_transform(X_test)
X_train  = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
#model = LinearRegression()
#model = BayesianRidge(compute_score=True)
#model = KNeighborsClassifier(n_neighbors = 3)
#model = DecisionTreeClassifier()
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='uniform'))
model.compile(loss='mae', optimizer='adam')
history = model.fit(X_train,Y_train,epochs=50,batch_size=128)
Y_pred = model.predict(X_test).astype(float)
Y_pred = Y_pred.reshape([Y_pred.shape[0]])
testHour = test_df['hour']
result = pd.DataFrame({
    'date' : dd,
    'hour' : testHour,
    'pm2.5' : Y_pred
})
#test.to_csv( 'titanic_pred_gradient_boosting.csv' , index = False )
result.to_csv(SCRIPT_PATH + "/pm25_lstm.csv", index=False)
