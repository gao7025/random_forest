# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:27:57 2020

@author: gao
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

class evaluateIndicator():

    def __init__(self):
        pass

    # 交叉验证
    def plot_cross_val(self, rf4, train_x, train_y, cv_num, path_out):
        warnings.filterwarnings("ignore")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        evaluate_vars = ['roc_auc', 'precision', 'recall', 'f1']
        fig = plt.figure()
        for plot_num in range(len(evaluate_vars)):
            ax1 = fig.add_subplot(2, 2, plot_num + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.3, hspace=0.5)
            try:
                scores = cross_val_score(rf4, train_x, train_y, cv=cv_num, scoring=evaluate_vars[plot_num])
            except ValueError:
                scores = np.zeros(10)
            plt.plot(range(10), scores)
            plt.xlabel('num of cv')
            plt.ylabel(evaluate_vars[plot_num])
            plt.xticks(np.arange(0, 10, 1), fontsize=6)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=8)
            plt.show()
            tt = 'plot of ' + str(evaluate_vars[plot_num])
            ax1.set_title(tt, fontsize=10)
        plt.savefig(path_out, bbox_inches='tight', dpi=300)  # bbox_inches='tight'帮助删除图片空白部分
        plt.show()

    def predict_result(self, rf4, train_x, train_y, test_x, test_y):
        # 拟合模型并预测
        rf4.fit(train_x,train_y)
        train_prob = rf4.predict_proba(train_x)
        test_prob = rf4.predict_proba(test_x)
        test_pred = rf4.predict(test_x)
        obj1, result_report = self.evaluate_result(test_y, test_pred)
        self.plot_roc(test_y, test_prob[:, 1])
        self.plot_pr(train_y, train_prob, test_y, test_prob)
        return obj1, result_report

    def evaluate_result(self, test_y, test_pred):
        # 评价指标1：混淆矩阵相关
        obj1 = confusion_matrix(test_y, test_pred)
        print('confusion_matrix\n', obj1)
        # 1.1综合评估
        result_report = classification_report(test_y,test_pred,digits=5)
        print(result_report)
        # 1.2分别查看
        print('accuracy:{}'.format(accuracy_score(test_y, test_pred)))
        print('precision:{}'.format(precision_score(test_y, test_pred)))
        print('recall:{}'.format(recall_score(test_y, test_pred)))
        print('f1-score:{}'.format(f1_score(test_y, test_pred)))
        return obj1, result_report

    # 评价指标2：roc
    def plot_roc(self, test_y, test_prob):
        fpr,tpr,threshold=roc_curve(test_y,test_prob)
        roc_auc=auc(fpr,tpr)
        # 画roc曲线
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('False Positive Rate',fontsize=12)
        plt.ylabel('True Positive Rate',fontsize=12)
        plt.title('Receiver operating characteristic example',fontsize=12)
        plt.legend(loc="lower right",fontsize=12)
        plt.show()

    # 评价指标3：pr曲线
    def plot_pr(self, train_y, train_prob,test_y,test_prob):
        def draw_pr(y_true, y_pred, label=None):
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            plt.plot(recall, precision, linewidth=2, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([-0.05, 1.05, -0.05, 1.05])
            plt.xlabel("Recall Rate")
            plt.ylabel("Precision Rate")
        draw_pr(train_y, train_prob, 'rf_train')
        draw_pr(test_y, test_prob, 'rf_test')
        plt.legend()
        plt.show()
