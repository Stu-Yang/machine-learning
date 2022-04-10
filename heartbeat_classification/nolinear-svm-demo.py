# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
# 使用make_moons函数产生交叉半圆形随机数据
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
# 先调用SVC类，设置好参数，此处使用三次多项式核函数再使用fit函数求解
clf = SVC(kernel='poly', degree=3, coef0=1, C=5)
clf.fit(X, y)
# 使用predict函数预测，并绘制等高线
x0s = np.linspace(-1.5, 2.5, 100)
x1s = np.linspace(-1, 1.5, 100)
x0, x1 = np.meshgrid(x0s, x1s)
X_pred = np.c_[x0.ravel(), x1.ravel()]
y_pred = clf.predict(X_pred).reshape(x0.shape)
plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.1)
# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.axis('equal')
plt.show()