import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model  # 引用 sklearn库，使用其中的线性回归模块

# 自己拟的数据
data = np.array([[51, 152], [53, 156], [54, 160], [55, 164],
                 [57, 168], [60, 172], [62, 176], [65, 180],
                 [69, 184], [72, 188]])

# 题目给的数据
# data = np.array([[58, 168], [55, 165], [85, 175], [75, 178], [70, 180],
#                  [78, 170], [77, 173], [83, 183], [80, 179], [85, 190]])

# 1. 实例化一个线性回归模型
regr = linear_model.LinearRegression()
# 2. 从data中提取出身高和体重，分别存放在X,Y变量中
x, y = data[:, 0].reshape(-1, 1), data[:, 1]
# 3. 在x,y上训练一个线性回归模型。如果训练顺利，则regr会存储训练完成之后的结果模型
regr.fit(x, y)
# 4. 画出身高与体重之间的分布关系
plt.scatter(x, y, color="red")
# 5. 画出已训练好的线条
plt.plot(x, regr.predict(x), color='blue')
# 6. 画x,y轴的标题
plt.xlabel('weight(kg)')
plt.ylabel('height(cm)')
# 7. 展示
plt.show()
# 8. 利用已训练好的模型去预测体重70公斤的人的身高 结果为176.81
y_predict = regr.predict(x)
print("MSE =", mean_squared_error(y, y_predict))
print("RMSE =", np.sqrt(mean_squared_error(y, y_predict)))
print("MAE =", mean_absolute_error(y, y_predict))
print('预测体重69公斤的身高为：%.2f' % regr.predict([[69]]))
