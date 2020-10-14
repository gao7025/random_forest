# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 09:27:57 2020

@author: gao
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from evaluateIndicator import evaluateIndicator
import warnings
warnings.filterwarnings("ignore")
path_out = 'E:/programGao/csdnProgram'

# 加载数据并划分数据集
data = pd.read_excel('E:/programGao/creditdata.xlsx','dataset')
x = data.drop(['uid','flag'],axis=1)
y = data['flag']
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=423)

# 0.查看初始效果
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(train_x,train_y)
y_predprob = rf0.predict_proba(test_x)[:,1]
print('oob_score : %f ,auc : %f' % (rf0.oob_score_,roc_auc_score(test_y, y_predprob)))


# 1.迭代次数
param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=20,
                                  min_samples_leaf=10,max_depth=5,max_features='sqrt' ,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(train_x,train_y)
means = gsearch1.cv_results_['mean_test_score']
std = gsearch1.cv_results_['std_test_score']
params = gsearch1.cv_results_['params']
for mean,std,param in zip(means,std,params):
    print("mean : %f std : %f %r" % (mean,std,param))
print('best_params :',gsearch1.best_params_)

# 2.最大深度和内部节点再划分所需最小样本数
param_test2 = {'max_depth':range(3,10,1), 'min_samples_split':range(10,101,10)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 40,
                                  min_samples_leaf=10,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(train_x,train_y)
means = gsearch2.cv_results_['mean_test_score']
std = gsearch2.cv_results_['std_test_score']
params = gsearch2.cv_results_['params']
for mean,std,param in zip(means,std,params):
    print("mean : %f std : %f %r" % (mean,std,param))
print('best_params :',gsearch2.best_params_)

# 查看此时结果
rf2 = RandomForestClassifier(n_estimators= 40, max_depth=5, min_samples_split=70,
                                  min_samples_leaf=10,max_features='sqrt' ,oob_score=True, random_state=10)
rf2.fit(train_x,train_y)
y_predprob2 = rf2.predict_proba(test_x)[:,1]
print('oob_score : %f ,auc : %f' % (rf2.oob_score_,roc_auc_score(test_y, y_predprob2)))

# 3.内部节点再划分所需最小样本数和叶子节点最少样本数
param_test3 = {'min_samples_split':range(10,101,10), 'min_samples_leaf':range(5,51,5)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 40, max_depth=5,
                                  max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(train_x,train_y)
means = gsearch3.cv_results_['mean_test_score']
std = gsearch3.cv_results_['std_test_score']
params = gsearch3.cv_results_['params']
for mean,std,param in zip(means,std,params):
    print("mean : %f std : %f %r" % (mean,std,param))
print('best_params :',gsearch3.best_params_)

# 4.最大特征数
param_test4 = {'max_features':range(3,11,1)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 40, max_depth=5, min_samples_split=70,
                                  min_samples_leaf=10 ,oob_score=True, random_state=10),
   param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(train_x,train_y)
means = gsearch4.cv_results_['mean_test_score']
std = gsearch4.cv_results_['std_test_score']
params = gsearch4.cv_results_['params']
for mean,std,param in zip(means,std,params):
    print("mean : %f std : %f %r" % (mean,std,param))
print('best_params :',gsearch4.best_params_)
# 查看此时结果
rf4 = RandomForestClassifier(n_estimators= 40, max_depth=5, min_samples_split=70,
                                  min_samples_leaf=10,max_features=3 ,oob_score=True, random_state=10)
rf4.fit(train_x,train_y)
y_predprob4 = rf4.predict_proba(test_x)[:,1]
print('oob_score : %f ,auc : %f' % (rf4.oob_score_,roc_auc_score(test_y, y_predprob4)))

# 查看交叉验证结果并进行预测与评估
el = evaluateIndicator()
el.plot_cross_val(rf4, train_x, train_y, 10, path_out)
obj1, result_report = el.predict_result(rf4, train_x, train_y, test_x, test_y)

