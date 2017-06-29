# -*- coding: utf-8 -*-
"""


http://qiita.com/yshi12/items/668e699ed70906f9868b

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

http://qiita.com/Lewuathe/items/09d07d3ff366e0dd6b24

http://qiita.com/ynakayama/items/ca3f5e9d762bbd50ad1f
http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes

https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf


http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
class_weight = {class_label01: weight01, class_label02: weight02, class_label03: weight03}


"""

#import numpy as np
#import matplotlib.pyplot as plt
##%matplotlib inline
#from pandas import DataFrame as DF

from sklearn import datasets
from sklearn.cross_validation import train_test_split # クロスバリデーション用

# データ用意
iris = datasets.load_iris()    # データロード
X = iris.data                  # 説明変数セット
Y = iris.target                # 目的変数セット
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25) # random_stateはseed値。

from sklearn import metrics       # 精度検証用


from sklearn.ensemble import RandomForestClassifier # Random Forest
model_rf = RandomForestClassifier(class_weight={0:1/37, 1:1/34, 2:1/41})
#model_rf = RandomForestClassifier(class_weight='auto')
#model_rf = RandomForestClassifier(class_weight='balanced')
#model_rf = RandomForestClassifier(class_weight=None)
model_rf.fit(X_train, Y_train)
predicted_rf = model_rf.predict(X_test) 
print("RandomForest",metrics.accuracy_score(Y_test, predicted_rf),"\n")




