# -*- coding: utf-8 -*-
"""


http://qiita.com/yshi12/items/668e699ed70906f9868b

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

http://qiita.com/Lewuathe/items/09d07d3ff366e0dd6b24

http://qiita.com/ynakayama/items/ca3f5e9d762bbd50ad1f
http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes

https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf

SVM,DT,RF,AB,XGB,KNN,LR,NB

"""

#import numpy as np
#import matplotlib.pyplot as plt
##%matplotlib inline


from sklearn import datasets
from sklearn.cross_validation import train_test_split # クロスバリデーション用

# データ用意
iris = datasets.load_iris()    # データロード
X = iris.data                  # 説明変数セット
Y = iris.target                # 目的変数セット
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25) # random_stateはseed値。

from sklearn import metrics       # 精度検証用

cond01 = 0
# SVM,DT,RF,AB,KNN  -> 0
# XGB  -> 1
# LR  -> 2
if cond01 == 0:
    from sklearn.svm import SVC # SVM用
    model_svm = SVC(); model_svm.fit(X_train, Y_train)
    predicted_svm = model_svm.predict(X_test) ; print("SVM",metrics.accuracy_score(Y_test, predicted_svm),"\n")
    
    from sklearn.tree import DecisionTreeClassifier #Decsion Tree
    model_dt = DecisionTreeClassifier(); model_dt.fit(X_train, Y_train)
    predicted_dt = model_dt.predict(X_test) ; print("DecisionTree",metrics.accuracy_score(Y_test, predicted_dt),"\n")

    from sklearn.tree import export_graphviz
    export_graphviz(model_dt, out_file="tree5.dot",
                         #feature_names=iris.feature_names,
                         #class_names=iris.target_names,
                         filled=True, rounded=True)
    import subprocess
    subprocess.check_call('dot -Tpng tree5.dot -o tree5.png', shell=True)



    from sklearn.ensemble import RandomForestClassifier # Random Forest
    model_rf = RandomForestClassifier(); model_rf.fit(X_train, Y_train)
    predicted_rf = model_rf.predict(X_test) ; print("RandomForest",metrics.accuracy_score(Y_test, predicted_rf),"\n")
    
    from sklearn.ensemble import AdaBoostClassifier # AdaBoost
    model_ab = AdaBoostClassifier(); model_ab.fit(X_train, Y_train)
    predicted_ab = model_ab.predict(X_test) ; print("AdaBoost",metrics.accuracy_score(Y_test, predicted_ab),"\n")
    
    from sklearn.neighbors import KNeighborsClassifier #K-NN
    model_knn = KNeighborsClassifier(); model_knn.fit(X_train, Y_train)
    predicted_knn = model_knn.predict(X_test) ; print("K-NN",metrics.accuracy_score(Y_test, predicted_knn),"\n")

if cond01 == 1:
    from xgboost import XGBClassifier  # XGBoost
    model_xgb = XGBClassifier(); model_xgb.fit(X_train, Y_train)
    predicted_xgb = model_xgb.predict(X_test); print("XGBoost",metrics.accuracy_score(Y_test, predicted_xgb),"\n")

if cond01 == 2:    
    from sklearn.linear_model import LogisticRegression # Logistic Regression
    model_lr = LogisticRegression(); model_lr.fit(X_train, Y_train)
    predicted_lr = model_lr.predict(X_test) ; print("LogisticRegression",metrics.accuracy_score(Y_test, predicted_lr),"\n")
    #aa = model_lr.coef_    
    
if cond01 == 3:
    from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
    model_nb = GaussianNB(); model_nb.fit(X_train, Y_train)
    predicted_nb = model_nb.predict(X_test) ; print("Gaussian Naive Bayes",metrics.accuracy_score(Y_test, predicted_nb),"\n")

if cond01 == 4:
    from sklearn.ensemble import GradientBoostingClassifier # GradientBoosting
    model_gb = GradientBoostingClassifier(); model_gb.fit(X_train, Y_train)
    predicted_gb = model_gb.predict(X_test) ; print("GradientBoosting",metrics.accuracy_score(Y_test, predicted_gb),"\n")

if cond01 == 5:
    from lightgbm import LGBMClassifier  # LightGBM
    model_lgbm = LGBMClassifier(); model_lgbm.fit(X_train, Y_train)
    predicted_lgbm = model_lgbm.predict(X_test); print("LightGBM",metrics.accuracy_score(Y_test, predicted_lgbm),"\n")



#
##http://myenigma.hatenablog.com/entry/2015/10/09/223629
#import seaborn as sns
#iris = sns.load_dataset("iris") #サンプルデータセット
##sns.pairplot(iris);
#sns.pairplot(iris,hue="species");
#sns.plt.savefig("iris.png")
#sns.plt.show()
#



#http://qiita.com/rikima/items/84c3797b721749a022e9
#cd ~/anaconda/bin
#sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))


