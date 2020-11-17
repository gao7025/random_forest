#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 17:48
# @Author  : gao

# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# ims = []
# fig = plt.figure()
# # for i in range(1, 10):
# #     im = plt.scatter(i + 1, i + 1).findobj()
# #     print(im)
# #     plt.draw()
# #     plt.pause(0.2)
# #     ims.append(im)
# a = list(range(1, 10))
# b = [each + 1 for each in a]
# c = [each + 2 for each in a]
#
# for i, j, k in zip(a, b, c):
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#     im = ax.scatter(a, b, k).findobj()
#     ax.set_xlim3d(0, 30)
#     ax.set_ylim3d(0, 30)
#     ax.set_zlim3d(0, 30)
#     # print(im)
#     # plt.draw()
#     # plt.pause(0.5)
#     ims.append(im)
# ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
# ani.save("test1.gif", writer='pillow')
#
# from PIL import Image
# import os
#
# im1 = Image.open(os.path.abspath('.') + "/test1.gif")
# im12 = im1.rotate(60)
# im12.show()




# import itertools
# tt = []
# for i in itertools.combinations_with_replacement(['BBC','aa','cc'], 2):
#     #print(i)
#     t = ''.join(i)
#     tt.append(t)
# # 输出 BC BD BE BF CD CE CF DE DF EF
# print('\n',set(tt))

# import copy    #实现list的深复制
#
# def Cnr(lst, n):
#     result = []
#     tmp = [0] * n
#     length = len(lst)
#     def next_num(li=0, ni=0):
#         if ni == n:
#             result.append(copy.copy(tmp))
#             return
#         for lj in range(li,length):
#             tmp[ni] = lst[lj]
#             next_num(lj+1, ni+1)
#     next_num()
#     return result
#
# mm = Cnr(['BBC','aa','cc', 'dd'], 3)
# print(mm)

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
#
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
# plt.savefig(r'C:\Users\will\Desktop\ttt.gif')
# plt.show()

# from sklearn import datasets
# import pandas as pd
# import numpy as np
# iris = datasets.load_iris()
# X = pd.DataFrame(iris.data)
# y = pd.DataFrame(iris.target)
# dd = pd.merge(X, y, left_index=True, right_index=True, how='left')
# dd.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
# print()

#ddata = pd.DataFrame()

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def visual(X,y):
    fig = plt.figure(1, figsize=(6, 4))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=labels.astype(np.float), edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('花瓣宽度')
    ax.set_ylabel('萼片长度')
    ax.set_zlabel('花瓣长度')
    ax.set_title("聚类(鸢尾花的三个特征数据)")
    ax.dist = 12
    plt.show()

if __name__=='__main__':
    np.random.seed(5)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    est = KMeans(n_clusters=3)
    est.fit(X)
    labels = est.labels_
    visual(X,y)

