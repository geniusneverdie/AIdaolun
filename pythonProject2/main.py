# 在这里测试GM(1,1)模型
import numpy as np
import math as mt
import matplotlib.pyplot as plt

X0=[236.822,251.056,265.29,279.524,293.758,307.992,322.226,336.46,350.694,364.928,379.162,393.396,407.63,421.864,436.098,450.332,464.566]
# 1.我们有一项初始序列X0
#.我们对其进行一次累加
X1 = [236.822]
add = X0[0] + X0[1]
X1.append(add)
i = 2
while i < len(X0):
    add = add + X0[i]
    X1.append(add)
    i += 1
print("X1", X1)

# 3.获得紧邻均值序列
Z = []
j = 1
while j < len(X1):
    num = (X1[j] + X1[j - 1]) / 2
    Z.append(num)
    j = j + 1
print("Z", Z)

# 4.最小二乘法计算
Y = []
x_i = 0
while x_i < len(X0) - 1:
    x_i += 1
    Y.append(X0[x_i])
Y = np.mat(Y).T
Y.reshape(-1, 1)
print("Y", Y)

B = []
b = 0
while b < len(Z):
    B.append(-Z[b])
    b += 1
print("B:", B)
B = np.mat(B)
B.reshape(-1, 1)
B = B.T
c = np.ones((len(B), 1))
B = np.hstack((B, c))
print("c", c)
print("b", B)

# 5.我们可以求出我们的参数
theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
a = theat[0][0]
b = theat[1][0]
print(b)
print(a)
did = 247.28524221/-0.03951812
print(theat)
print(type(theat))

# 6.生成我们的预测模型
F = []
F.append(X0[0])
k = 1
while k < len(X0):
    F.append((X0[0] - did) * mt.exp(-a * k) + did)
    k += 1
print("F", F)

# 7.两者做差得到预测序列
G = []
G.append(X0[0])
g = 1
while g < len(X0):
    print(g)
    G.append(F[g] - F[g - 1])
    g += 1
print(F)

r = range(17)
t = list(r)

X0 = np.array(X0)
G = np.array(G)
e = X0 - G;
q = e / X0;  # 相对误差
s1 = np.var(X0)  # 方差
s2 = np.var(e)

c = s2 / s1  # 方差的比值

p = 0;  # 小误差概率

for s in range(len(e)):
    if (abs(e[s]) < 0.6745 * s1):
        p = p + 1;
P = p / len(e)
print(c)
print(P)
print(G)

plt.plot(t, X0,color='r',linestyle="--",label='true')
plt.plot(t, G,color='b',linestyle="--",label="predict")
plt.legend(loc='upper right')
plt.show()



