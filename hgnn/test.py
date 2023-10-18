from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 假设 X 和 y 是我们的数据
X = np.random.rand(100, 10)
y = np.random.rand(100)

kf = KFold(n_splits=5, random_state=42, shuffle=True)


print(y)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # 计算并打印每一折的均方误差
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
