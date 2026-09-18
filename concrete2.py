import numpy as np
import matplotlib.pyplot as plt

# 构造模拟数据集，代替UCI混凝土数据，不需要联网
np.random.seed(42)
n = 1030
X = np.random.randn(n,8)
# 真实权重
true_w = np.array([30, 8, 5, 2, -4, -1, 1, 3, -2]).reshape(-1,1)
# 增加常数项
X_b = np.hstack([np.ones((n,1)), X])
y = X_b @ true_w + np.random.randn(n,1)*1.2

# 划分训练集、验证集
split = int(n * 0.8)
X_train = X_b[:split]
y_train = y[:split]
X_val = X_b[split:]
y_val = y[split:]

# MSE损失函数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# 梯度下降
def gradient_descent(X, y, lr=0.01, epoch=2000):
    n_feature = X.shape[1]
    w = np.zeros((n_feature,1))
    train_loss_list = []
    val_loss_list = []
    
    for i in range(epoch):
        y_pred_train = X @ w
        grad = (2 / len(X)) * X.T @ (y_pred_train - y)
        w = w - lr * grad
        
        loss_train = mse_loss(X_train @ w, y_train)
        loss_val = mse_loss(X_val @ w, y_val)
        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)
    return w, train_loss_list, val_loss_list

# 训练
w, train_loss, val_loss = gradient_descent(X_train,y_train,lr=0.01,epoch=2000)

print("训练集最终损失：", train_loss[-1])
print("验证集最终损失：", val_loss[-1])

# 画loss曲线
plt.figure(figsize=(8,4))
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.xlabel("迭代次数")
plt.ylabel("MSE损失")
plt.legend()
plt.title("Loss变化曲线")
plt.show()

# 真实vs预测散点图
y_pred_val = X_val @ w
plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred_val)
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.plot([y_val.min(),y_val.max()],[y_val.min(),y_val.max()], color='red')
plt.title("真实值 vs 预测值")
plt.show()
