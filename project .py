
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl 
from sklearn import preprocessing 
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[15]:

house = pd.read_csv("D:\科目文件\大数据导论\midterm\kc_house_data.csv")
house.head(10)


# In[16]:

#数据可视化，散点图，箱体图绘制
house.plot(kind="scatter", x="bedrooms", y="price")
sns.plt.show()


# In[17]:

sns.boxplot(x="bedrooms", y="price", data=house)
sns.plt.show()


# In[19]:

house.plot(kind="scatter", x="bathrooms", y="price")
sns.plt.show()


# In[20]:

sns.boxplot(x="bathrooms", y="price", data=house)
sns.plt.show()


# In[21]:

house.plot(kind="scatter", x="sqft_living", y="price")
sns.plt.show()


# In[38]:

house.plot(kind="scatter", x="sqft_lot", y="price")
sns.plt.show()


# In[25]:

sns.boxplot(x="view", y="price", data=house)
sns.plt.show()


# In[28]:

sns.boxplot(x="yr_renovated", y="price", data=house)
sns.plt.show()


# In[11]:

house.plot(kind="scatter", x="date", y="price")
sns.plt.show()


# In[12]:

#数据中缺失值的查找
house_value_ravel=house.values.ravel()
print('数据中缺失值个数',len(house_value_ravel[house_value_ravel==np.nan]))


# In[14]:

#查看样本中数据分布
print('数据中各类别样本分布：')
print(house['bedrooms'].value_counts().sort_index())


# In[15]:

#查看样本中数据分布
print('数据中各类别样本分布：')
print(house['bathrooms'].value_counts().sort_index())


# In[16]:

#房价直方图
house['price'].describe()
pl.hist(house["price"])
pl.xlabel('price')
pl.ylabel('count')
pl.show()


# In[6]:

#0-1标准化
minmax_scale=preprocessing.MinMaxScaler().fit(house[["date","bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]])
df_minmax=minmax_scale.transform(house[["date","bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]])


# In[7]:

#打印标准化后最小值和最大值
print('Min-value after 0-1 scaling:\nbedrooms={:.2f},bathrooms={:.2f},sqft_living={:.2f},sqft_lot={:.2f},floors={:.2f},waterfront={:.2f},view={:.2f},condition={:.2f},grade={:.2f},sqft_above={:.2f},sqrt_basement={:.2f},yr_built={:.2f},yr_renovated={:.2f},zipcode={:.2f},lat={:.2f},long={:.2f},sqft_living15={:.2f},sqrt_lot15={:.2f}'.format(df_minmax[:,0].min(),df_minmax[:,1].min(),df_minmax[:,2].min(),df_minmax[:,3].min(),df_minmax[:,4].min(),df_minmax[:,5].min(),df_minmax[:,6].min(),df_minmax[:,7].min(),df_minmax[:,8].min(),df_minmax[:,9].min(),df_minmax[:,10].min(),df_minmax[:,11].min(),df_minmax[:,12].min(),df_minmax[:,13].min(),df_minmax[:,14].min(),df_minmax[:,15].min(),df_minmax[:,16].min(),df_minmax[:,17].min()))
print('\nMax-value after 0-1 scaling:\nbedrooms={:.2f},bathrooms={:.2f},sqft_living={:.2f},sqft_lot={:.2f},floors={:.2f},waterfront={:.2f},view={:.2f},condition={:.2f},grade={:.2f},sqft_above={:.2f},sqrt_basement={:.2f},yr_built={:.2f},yr_renovated={:.2f},zipcode={:.2f},lat={:.2f},long={:.2f},sqft_living15={:.2f},sqrt_lot15={:.2f}'.format(df_minmax[:,0].max(),df_minmax[:,1].max(),df_minmax[:,2].max(),df_minmax[:,3].max(),df_minmax[:,4].max(),df_minmax[:,5].max(),df_minmax[:,6].max(),df_minmax[:,7].max(),df_minmax[:,8].max(),df_minmax[:,9].max(),df_minmax[:,10].max(),df_minmax[:,11].max(),df_minmax[:,12].max(),df_minmax[:,13].max(),df_minmax[:,14].max(),df_minmax[:,15].max(),df_minmax[:,16].max(),df_minmax[:,17].max()))


# In[3]:

#打印相关系数矩阵
house[[ "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]].corr()


# In[9]:

V=pd.DataFrame(df_minmax) 


# In[10]:

V.head(15) #查看标准化后结果


# In[3]:

#去掉id和price开始模型的构建
house = house.drop(['id'], axis = 1)
house1=house.drop(['price'],axis=1)
house2 = house.drop(['price'],axis=1)
#增加样本特征
a1 = house2['grade']
b1=[pow(a1[i],4.7) for i in range(len(a1))]
house2['g2']=pd.Series(b1,index = house2.index)

a2 = house2['bathrooms']
b2=[pow(a2[i],1.0000000001) for i in range(len(a2))]
house2['bath2']=pd.Series(b2,index = house2.index)

a3 = house2['lat']
b3=[pow(a3[i],8.6)for i in range(len(a3))]
house2['lat2']=pd.Series(b3,index = house2.index)

a4 = house2['sqft_lot']
b4=[pow(1/a4[i],0.3)for i in range(len(a4))]
house2['sqflot2']=pd.Series(b4,index = house2.index)

a5 = house2['yr_built']
b5=[pow(a5[i],4.1) for i in range(len(a5))]
house2['yrblt2']=pd.Series(b5,index = house2.index)


# In[5]:

#未增加样本特征的训练集，测试集的划分
train_x = house1.iloc[0:15130,]
test_x = house1.iloc[15130:21613,]
Y=house['price']
train_y = Y.iloc[0:15130]
test_y = Y.iloc[15130:21613]


# In[6]:

#增加了样本特征的训练集
train_x1 = house2.iloc[0:15130,]
test_x1 = house2.iloc[15130:21613,]


# In[6]:

#利用线性回归模型开始对房价进行预测
regr = linear_model.LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False)
regr.fit(train_x1,train_y)
pred = regr.predict(test_x1)
#评价模型的建立
r2 = r2_score(test_y, pred)
mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y,pred)
rmse = mse**0.5
print("R2:",r2)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("Root mean squared error:",rmse)


# In[7]:

#利用Lasso回归模型开始对房价进行预测
regr = linear_model.Lasso()
regr.fit(train_x1,train_y)
pred = regr.predict(test_x1)
#评价模型的建立
r2 = r2_score(test_y, pred)
mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y,pred)
rmse = mse**0.5
print("R2:",r2)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("Root mean squared error:",rmse)


# In[8]:

#利用岭回归模型开始对房价进行预测
regr = linear_model.Ridge(alpha=0.5)
regr.fit(train_x1,train_y)
pred = regr.predict(test_x1)
#评价模型的建立
r2 = r2_score(test_y, pred)
mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y,pred)
rmse = mse**0.5
print("R2:",r2)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("Root mean squared error:",rmse)


# In[11]:

#利用随机森林对房价进行预测
clf=RandomForestClassifier(n_estimators = 42)
clf.fit(train_x,train_y)
pred=clf.predict(test_x)

#评价模型的建立
r2 = r2_score(test_y, pred)
mae = mean_absolute_error(test_y, pred)
mse = mean_squared_error(test_y,pred)
rmse = mse**0.5
print("R2:",r2)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("Root mean squared error:",rmse)


# In[ ]:



