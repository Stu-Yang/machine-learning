#SVM demo

# 引入必要的库
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# 使用正态分布产生随机数据
x = np.r_[np.random.randn(20,2) - [2,2], np.random.randn(20,2) + [2,2]]
y = [0]*20 + [1]*20

# 先调用SVC类，设置好参数，再使用fit函数求解，使用predict函数预测
clf = SVC(kernel = 'linear')
clf.fit(x, y)
clf.predict([[2,2], [-2,-2]])

#获取参数w,b，以及支持向量
(w, b, sv) = (clf.coef_[0], clf.intercept_[0], clf.support_vectors_)

# 绘图部分
x1 = np.linspace(-5, 5)
x2 = -(w[0] * x1 + b) / w[1]
x2up = -(w[0] * x1 + b - 1) / w[1]
x2down = -(w[0] * x1 + b + 1) / w[1]
plt.figure(figsize=(8, 8))
plt.plot(x1, x2)
plt.plot(x1, x2up, linestyle="--")
plt.plot(x1, x2down, linestyle="--")
plt.scatter(sv[:, 0], sv[:, 1], s=80)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.axis('equal')
plt.show()