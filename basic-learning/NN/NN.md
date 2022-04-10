### 简单全连接前馈神经网络的构建

神经网络(Neural Network)重要组件：神经元(Neuron)

假设下面的神经元完成这样的一个函数：$y=f(x1*w1+x2*w2+b)$，其中$f(x)$是激活函数，这里采用sigmod函数

```python
import numpy as np

# 激活函数f(x) = 1 / (1 + e^(-x))
def sigmod(x):
    return 1 / (1 + np.exp(-x))

# 神经元类
class Neuron:
    def __init__(self, weight, bias):
        self.weights = weight
        self.bias = bias
    def feedforward(self, inputs):
        sum = np.dot(self.weights, inputs) + self.bias  #利用numpy中向量点积运算
        return sigmod(sum)
```

```python
# 下面是一个例子
weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)
x = np.array([2,3])
print(n.feedforward(x))
```
```python
>>> 0.9990889488055994
```


神经网络由神经元组成，下面介绍如何利用神经元组成全连接前馈神经网络。

神经网络由1个输入层、n个隐藏层和1个输出层构成。假设每个神经元定义和上述的神经元类完全相同，神经网络有1个输入层(x1,x2)、1个隐藏层(h1,h2)和1个输出层(o1)构成
其中$y=f(o1), o1=f(w5*h1 + w6*h2 + b3), h1=f(w1*x1 + w2*x2 + b1), h2=f(w3*x1 + w4*x2 + b2)$
为了简化，我们假设$w=[0,1], b=0$。


```python
# 神经网络类
class Neural_Network:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias)  # 这里的神经网络有三个神经元：隐藏层的h1,h2、输出层的o1
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1
```

```python
# 下面是一个实际的例子
NN = Neural_Network()
x = np.array([2, 3])
print(NN.feedforward(x))
```
```python
>>> 0.7216325609518421
```

有了上面的基础之后，下面看看实际的例子，同时我们也要开始进行模型训练阶段了。假设训练数据如下:

|  Name   | Height | weight | F/M  |
| :-----: | :----: | :----: | :--: |
|  Alice  |  133   |   65   |  F   |
|   Bob   |  162   |   72   |  M   |
| Charlie |  152   |   70   |  M   |
|  Diana  |  120   |   60   |  F   |

对上述数据进行预处理，每个人的Height和weight分别减去所有人的均值，另外用0表示F、1表示M
|  Name   | Height | weight | F/M  |
| :-----: | :----: | :----: | :--: |
|  Alice  |   -2   |   -1   |  1   |
|   Bob   |   25   |   6    |  0   |
| Charlie |   17   |   4    |  0   |
|  Diana  |  -15   |   -6   |  1   |


```python
# 下面定义损失函数，用平均方差损失MSE表示损失函数
def mse_loss(y_ture, y_pred):
    return ( (y_ture - y_pred)**2 ).mean()
```


万事俱备，现在可以训练模型了。我们的目标是：根据数据求解w和b的值，使得MSE最小，我们在这里利用反向传播算法进行求解，下面是整个的求解算法

```python
import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # 权重，Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # 截距项，Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

```
```python
Epoch 0 loss: 0.355
Epoch 100 loss: 0.019
Epoch 200 loss: 0.009
Epoch 300 loss: 0.006
Epoch 400 loss: 0.004
Epoch 500 loss: 0.003
Epoch 600 loss: 0.003
Epoch 700 loss: 0.002
Epoch 800 loss: 0.002
Epoch 900 loss: 0.002
```

现在我们可以用这个神经网络来预测性别了：
```python
# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) 
print("Frank: %.3f" % network.feedforward(frank)) 
```
```python
Emily: 0.946
Frank: 0.039
```
