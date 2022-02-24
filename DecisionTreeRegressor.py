# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:05:33 2022

@author: passi
"""

import pandas as pd
# from supervised_learning import supervised_learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
# data = pd.read_csv('Boston Housing dataset.txt', delimiter = "\t", header = None)
from sklearn.metrics import r2_score

random_seed = 42
test_data_size = 0.2

def get_xy():
    with open('Boston Housing dataset.txt') as f:
        mylist = f.read().splitlines()
        
    x1 = mylist[::2]
    x1 = [e.split() for e in x1]
    df1 = pd.DataFrame(x1)
    x2 = mylist[1::2]
    x2 = [e.split() for e in x2]
    df2 = pd.DataFrame(x2)
    
    df = pd.concat([df1, df2], axis = 1)
    
    y = df.iloc[:,-1]
    X = df.iloc[:,:-1]
    return X, y


def data_preprocess(X, y):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_data_size, random_state = random_seed)
    return X_train, X_test, y_train, y_test


def perform():
    X, y = get_xy()
    X_train, X_test, y_train, y_test = data_preprocess(X, y)    
    regressor = DecisionTreeRegressor(random_state = random_seed, max_depth = 5)
    # cv_score = cross_val_score(regressor, X, y, cv=10, scoring = 'r2').mean()
    regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    r2_score_train = r2_score(y_train, regressor.predict(X_train))
    r2_score_test = r2_score(y_test, regressor.predict(X_test))
    print(r2_score_train, r2_score_test)
    
perform()

