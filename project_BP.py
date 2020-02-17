

#BP的python部分代码

import pandas as pd
import numpy as np
 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#BP数据导入
test_y = pd.read_csv("D:\kc_house_data3.csv")

pred = np.array(pd.read_csv("D:\kc_house_data2.csv"))
#BP评价指标建立
r2 = r2_score(test_y, pred)
mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y,pred)
rmse = mse**0.5
print("R2:",r2)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("Root mean squared error:",rmse)
