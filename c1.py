import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, alpha, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
    return theta

def predict(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

def accuracy(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yp >= 0.5:
            prediction = 1
        else:
            prediction = 0
        if yt == prediction:
            correct += 1
    return correct / len(y_true)

# 从Excel文件读取数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df[['密度', '含糖量']].values
    y = df['好瓜'].map({'是': 1, '否': 0}).values
    return X, y

# 添加偏置项
def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

# 主函数
def main():
    file_path = "work1data.xlsx"
    X, y = load_data(file_path)

    # 添加偏置项
    X = add_bias(X)

    # 划分训练集和测试集
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    train_indices, test_indices = indices[:split], indices[split:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # 使用梯度下降求解模型参数
    alpha = 0.01
    num_iterations = 1000
    theta = gradient_descent(X_train, y_train, alpha, num_iterations)

    # 在测试集上进行预测
    y_pred_proba = predict(X_test, theta)
    y_pred = np.round(y_pred_proba)

    # 计算准确率
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
