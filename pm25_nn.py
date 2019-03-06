import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from sklearn.preprocessing import MinMaxScaler

#get training data
#with open('train.csv') as csvfile:
#    readCSV = csv.reader(csvfile)
#    for row in readCSV:
#        print(row)
# get titanic & test csv files as a DataFrame
# get titanic & test csv files as a DataFrame
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
train_df = pd.read_csv(SCRIPT_PATH + "/train.csv")
test_df = pd.read_csv(SCRIPT_PATH + "/test.csv")
dd = test_df['day']
testHour = test_df['hour']
combine = [train_df, test_df]

#change day from year/month/day to sorted by season
train_df['time'] = train_df['day']
for i in range(train_df.shape[0]):
    year,month,day = (train_df['day'][i]).split("/")
    month = int(month)
    if(month>=1)and(month<=3):
        train_df.at[i,'time'] = 1
    elif (month>=4)and(month<=6):
        train_df.at[i,'time'] = 2
    elif (month>=7)and(month<=9):
        train_df.at[i,'time'] = 3
    else:
        train_df.at[i,'time'] = 4

test_df['time'] = test_df['day']     
for i in range(test_df.shape[0]):
    year,month,day = (test_df['day'][i]).split("/")
    month = int(month)
    if(month>=1)and(month<=3):
        test_df.at[i,'time'] = 1
    elif (month>=4)and(month<=6):
        test_df.at[i,'time'] = 2
    elif (month>=7)and(month<=9):
        test_df.at[i,'time'] = 3
    else:
        test_df.at[i,'time'] = 4

train_df = train_df.drop(['day'], axis=1)
test_df = test_df.drop(['day'], axis=1)
combine = [train_df, test_df]
scaler = MinMaxScaler()
    
for dataset in combine:
    dataset['wind'] = 0
    dataset.loc[ dataset['wind_ne'] == 1, 'wind'] = 1
    dataset.loc[ dataset['wind_se'] == 1, 'wind'] = 2
    dataset.loc[ dataset['wind_cv'] == 1, 'wind'] = 4
    dataset.loc[ dataset['wind_nw'] == 1, 'wind'] = 3
        
train_df = train_df.drop(['wind_ne', 'wind_nw', 'wind_cv', 'wind_se'], axis=1)
test_df = test_df.drop(['wind_ne', 'wind_nw', 'wind_cv', 'wind_se'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("pm2.5", axis=1)
Y_train = train_df["pm2.5"]
X_train = scaler.fit_transform(X_train)
dim = train_df.shape

#Session
sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 9])
W1 = tf.Variable(tf.zeros([9,20]))
b1 = tf.Variable(tf.zeros([20]))
Z3 = tf.nn.tanh(tf.matmul(x,W1)+b1)
W2 = tf.Variable(tf.zeros([20, 1]))
b2 = tf.Variable(tf.zeros([1]))
y = tf.matmul(Z3,W2)+b2
y_ = tf.placeholder(tf.float32, [None])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.squared_difference(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
batch_size = 32
num_batch = dim[0]//batch_size
init_op = tf.global_variables_initializer()
with sess.as_default():
    sess.run(init_op)
    for i in range(num_batch):
        batch_xt = X_train[i*batch_size:((i+1)*batch_size-1)][:]
        batch_yt = Y_train[i*batch_size:((i+1)*batch_size-1)][:]
        train_step.run({x:batch_xt, y_:batch_yt})
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #tess = tf.argmax(y,1)
    #print(tess.eval({x:batch_xt}))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(accuracy.eval({x:train_x, y_:train_y_}))
    X_test = test_df.copy()
    X_test = scaler.transform(X_test)
    Y_pred = sess.run(y, feed_dict={x:X_test})
    Y_pred = Y_pred.reshape([Y_pred.shape[0]])
    result = pd.DataFrame({
        'date' : dd,
        'hour' : testHour,
        'pm2.5' : Y_pred
        })
    result.to_csv(SCRIPT_PATH + "/submission_nn.csv", index=False)
    
