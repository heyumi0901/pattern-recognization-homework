# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
from adacost import AdaCostClassifier
from sklearn.ensemble import AdaBoostClassifier
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_creditcard_data(): # 读取数据，并将正例标记为1，负例标记为-1
    df = pd.read_csv('creditcard_train.csv')

    df.loc[df.Class == 1, 'Class'] = -1
    df.loc[df.Class == 0, 'Class'] = 1
    print(df.shape)
    print(df.Class.value_counts())
    return df.drop('Class', axis=1), df['Class']



if __name__ == '__main__':
     # X_train, y_train = load_creditcard_data()＃输出预测数据时用
     X, y= load_creditcard_data()
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #分割数据集，将部分训练集用作测试集

     # X_test= pd.read_csv('creditcard_test.csv' ＃输出预测数据时用
     # ZZ=load_creditcardtest_data()

     clf = AdaCostClassifier(n_estimators=100)
     clf.fit(X_train, y_train)
     y_pred = clf.predict(X_test)
     print(pd.Series(y_pred).value_counts())
     print('recall=',recall_score(y_test, y_pred, pos_label=-1),'  '
           'precision=',precision_score(y_test, y_pred, pos_label=-1),'  '
           'f1_score=',f1_score(y_test, y_pred, pos_label=-1), )

     # answer = pd.read_csv(open('sample_Submission.csv')) # 输出预测csv
     # for i in range(y_pred.shape[0]):
     #     predict = y_pred[i]
     #     if predict==1:
     #         answer.loc[i, "class"] = "1"
     #     else:
     #         answer.loc[i, "class"] = "0"
     # answer.to_csv('submission1.csv', index=False)


