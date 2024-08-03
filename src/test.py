from sklearn.model_selection import train_test_split
import numpy as np

# 假设我们有一些数据和目标变量
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([1, 2, 3, 4, 5, 6])

# 指定测试集的比例，例如20%
test_size = 0.2

# 使用train_test_split函数，设置shuffle=False
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False, random_state=None
)

# 打印结果
print("Training set features:\n", X_train)
print("Test set features:\n", X_test)
print("Training set labels:\n", y_train)
print("Test set labels:\n", y_test)