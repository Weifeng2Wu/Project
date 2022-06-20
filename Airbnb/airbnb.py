import sys
import pandas as pd

pd.set_option("display.max_column", 1000)
pd.set_option("display.width", 1000)

import glob
import matplotlib  # collection of functions for scientific and publication-ready visualization
import numpy as np  # foundational package for scientific computing
import scipy as sp  # collection of functions for scientific computing and advance mathematics
import sklearn  # collection of machine learning algorithms

# print("python version : {}".format(sys.version))
# print("python version : {}".format(sp.__version__))

import warnings

warnings.filterwarnings("ignore")

# Visualize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import autocorrelation_plot

# print("seaborn version : {}".format(sns.__version__))

mpl.style.use("ggplot")  # 折线图
sns.set_style("white")
pylab.rcParams['figure.figsize'] = 12, 8  # 四格图

airbnb = pd.read_csv("./input/AB_NYC_2019.csv")

# print("The shape of the data is (row,column)" + str(airbnb.shape))
# print(airbnb.info())
# print(airbnb.head())

# check missing values
import missingno as msno

# msno.matrix(airbnb)
# plt.show()

# print("data column with null values:",airbnb.isnull().sum(), sep="\n")
# default values replace null values
airbnb["reviews_per_month"].fillna(value=0, inplace=True)  # 直接替换当前表
airbnb.drop(['id', 'host_name', 'last_review'], axis=1, inplace=True)  # drop column
print(airbnb.head())


def filters(airbnb, col):
    Q1 = airbnb[col].quantile(0.25)
    Q3 = airbnb[col].quantile(0.75)
    IQR = Q3 - Q1
    filter = ((airbnb[col] >= Q1 - 1.5 * IQR) & (airbnb[col] <= Q3 + 1.5 * IQR))
    return airbnb.loc[filter]


airbnb_new = filters(airbnb, 'price')
airbnb_new = filters(airbnb_new, "number_of_reviews")
airbnb_new = filters(airbnb_new, 'reviews_per_month')

# check outliers(boxplot)
# plt.figure(figsize=(15, 10))
# plt.style.use('seaborn-white')
# ax = plt.subplot(221)  # 2*2 左上
# plt.boxplot(airbnb_new['number_of_reviews'])
# ax.set_title("Number of Review")
# ax = plt.subplot(222)  # 2*2 右上
# plt.boxplot(airbnb_new['price'])
# ax.set_title("Price")
# ax = plt.subplot(223)  # 2*2 左下
# plt.boxplot(airbnb_new['availability_365'])
# ax.set_title("Availability")
# ax = plt.subplot(224)  # 2*2 右下
# plt.boxplot(airbnb_new['reviews_per_month'])
# ax.set_title("Reviews_per_month")
# plt.show()

# 柱状图
# plt.figure(figsize=(15,7))
# plt.style.use('seaborn-white')
# plt.subplot(221)
# sns.countplot(x='neighbourhood_group',data=airbnb_new,palette="Greens_d",
#               order=airbnb_new.neighbourhood_group.value_counts().index)
# fig = plt.gcf()
# fig.set_size_inches(8,8)
#
# plt.subplot(222)
# ax = sns.countplot(x='neighbourhood',data=airbnb_new,palette="Greens_d",
#               order=airbnb_new.neighbourhood.value_counts().iloc[:10].index)
# fig = plt.gcf()
# fig.set_size_inches(8,8)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha='right')#右旋转40度
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
#
# plt.subplot(223)
# sns.countplot(x='room_type',data=airbnb_new,palette="Greens_d",
#               order=airbnb_new.room_type.value_counts().index)
# fig = plt.gcf()
# fig.set_size_inches(8,8)
#
# plt.subplot(224)
# ax = sns.countplot(x='host_id',data=airbnb_new,palette="Greens_d",
#               order=airbnb_new.host_id.value_counts().iloc[:10].index)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha='right')
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
# fig = plt.gcf()
# fig.set_size_inches(8,8)
# plt.show()

# 分布图
# plt.figure(figsize=(15, 7))
# plt.style.use("seaborn-white")
# plt.subplot(221)
# sns.distplot(airbnb_new["price"])
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# plt.subplot(222)
# sns.distplot(airbnb_new["reviews_per_month"])
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# plt.subplot(223)
# sns.distplot(airbnb_new["number_of_reviews"])
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# plt.subplot(224)
# sns.distplot(airbnb_new["availability_365"])
# fig = plt.gcf()
# fig.set_size_inches(10, 10)
# plt. show()


# 箱型图
# plt.style.use('ggplot')
# plt.rcParams["figure.figsize"] = (16, 8)
# ax = sns.boxplot(x=airbnb_new['neighbourhood_group'], y=airbnb_new['price'], data=airbnb_new, palette='Set3')#价格与所在区域的关系
# ax.set_xlabel(xlabel='localtion', fontsize=20)
# ax.set_ylabel(ylabel='price', fontsize=20)
# ax.set_title(label='Distribution of prices accross location', fontsize=30)
# plt.xticks(rotation=90)
# plt.show()

# 饼状图
# plt.style.use('seaborn-white')
# f, ax = plt.subplots(1,2,figsize=(18,8))
# airbnb_new['room_type'].value_counts().plot.pie(explode=[0,0.05,0],autopct="%1.1f",ax=ax[0],shadow=True)
# ax[0].set_title("Room Type")
# ax[0].set_ylabel("Room Type Share")
# sns.countplot(x="room_type",hue="neighbourhood_group",data=airbnb_new)
# ax[1].set_title('Room Type for neighbouthood group')
# plt.show()

# 热力图
# f, ax = plt.subplots(figsize=(10, 10))
# sns.heatmap(airbnb_new.corr(), annot=True, linewidths=0.5, linecolor='black', fmt='.1f', ax=ax)
# plt.show()

# 经纬
plt.figure(figsize=(15, 15))
plt.style.use('seaborn-white')
plt.subplot(221)
sns.scatterplot(x='latitude', y='longitude', hue='neighbourhood_group', data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10, 10)


plt.subplot(222)
sns.scatterplot(x='latitude', y='longitude', hue='room_type', data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10, 10)

plt.subplot(223)
sns.scatterplot(x='latitude', y='longitude', hue='price', data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10, 10)

plt.subplot(224)
sns.scatterplot(x='latitude', y='longitude', hue='availability_365', data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.show()

# 地图上显示热力图
# import geopandas
# import math
# import folium
# from folium import Choropleth, Circle, Marker
# from folium.plugins import HeatMap, MarkerCluster
#
# m_1 = folium.Map(location=[40.7128, -74.0060], title='cartodbpositron',zoom_start=12)
#
# HeatMap(data=airbnb_new[['latitude', 'longitude']], radius=10).add_to(m_1)
#
# import webbrowser
# output_file = "map.html"
# m_1.save(output_file)
# webbrowser.open(output_file, new=2)
#
#
