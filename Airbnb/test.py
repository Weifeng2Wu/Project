#load packages
import sys
import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
# pd.set_option("display.max_colwidth", 1000)

import glob
import matplotlib #collection of functions for scientific and publication-ready visualization
import numpy as np #foundational package for scientific computing
import scipy as sp #collection of functions for scientific computing and advance mathematics

import sklearn #collection of machine learning algorithms
print("Python version: {}". format(sys.version))
print("pandas version: {}". format(pd.__version__))
print("matplotlib version: {}". format(matplotlib.__version__))
print("NumPy version: {}". format(np.__version__))
print("SciPy version: {}". format(sp.__version__))

print("scikit-learn version: {}". format(sklearn.__version__))
print('-'*30)
print(glob.glob("./input/*"))


#ignore warnings
import warnings
warnings.filterwarnings('ignore')


#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import autocorrelation_plot

#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser
# %matplotlib inline

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

#Loading the single csv file to a variable named 'airbnb'
airbnb=pd.read_csv("./input/AB_NYC_2019.csv")

#Lets look at a glimpse of table
# !! open cvs to glimpse
#print(airbnb.head())

print("The shape of the  data is (row, column):" + str(airbnb.shape))
print(airbnb.info())

import missingno as msno
msno.matrix(airbnb)
# plt.show()

print('Data columns with null values:',airbnb.isnull().sum(), sep='\n')


airbnb['reviews_per_month'].fillna(value=0, inplace=True)
print('Reviews_per_month column with null values:',airbnb['reviews_per_month'].isnull().sum(), sep = '\n')
airbnb.drop(['id','host_name','last_review'], axis = 1,inplace=True)
print(airbnb.head())

plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(airbnb['number_of_reviews'])
ax.set_title('Numer of Reviews')
ax=plt.subplot(222)
plt.boxplot(airbnb['price'])
ax.set_title('Price')
ax=plt.subplot(223)
plt.boxplot(airbnb['availability_365'])
ax.set_title('availability_365')
ax=plt.subplot(224)
plt.boxplot(airbnb['reviews_per_month'])
ax.set_title('reviews_per_month')
# plt.show()

def filters(airbnb, x):
    Q1 = airbnb[x].quantile(0.25)
    Q3 = airbnb[x].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range.

    filter = (airbnb[x] >= Q1 - 1.5 * IQR) & (airbnb[x] <= Q3 + 1.5 *IQR)
    return airbnb.loc[filter]

# airbnb = filters(airbnb, "price")

Q1 = airbnb['price'].quantile(0.25)
Q3 = airbnb['price'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range.

filter = (airbnb['price'] >= Q1 - 1.5 * IQR) & (airbnb['price'] <= Q3 + 1.5 *IQR)
airbnb1=airbnb.loc[filter]

Q1 = airbnb1['number_of_reviews'].quantile(0.25)
Q3 = airbnb1['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range.

filter = (airbnb1['number_of_reviews'] >= Q1 - 1.5 * IQR) & (airbnb1['number_of_reviews'] <= Q3 + 1.5 *IQR)
airbnb2=airbnb1.loc[filter]


Q1 = airbnb2['reviews_per_month'].quantile(0.25)
Q3 = airbnb2['reviews_per_month'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range.

filter = (airbnb2['reviews_per_month'] >= Q1 - 1.5 * IQR) & (airbnb2['reviews_per_month'] <= Q3 + 1.5 *IQR)
airbnb_new=airbnb2.loc[filter]

plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(airbnb_new['number_of_reviews'])
ax.set_title('Numer of Reviews')
ax=plt.subplot(222)
plt.boxplot(airbnb_new['price'])
ax.set_title('Price')
ax=plt.subplot(223)
plt.boxplot(airbnb_new['availability_365'])
ax.set_title('availability_365')
ax=plt.subplot(224)
plt.boxplot(airbnb_new['reviews_per_month'])
ax.set_title('reviews_per_month')



plt.figure(figsize = (15, 7))
plt.style.use('seaborn-white')
#Neighbourhood group
plt.subplot(2, 2, 1)
sns.countplot(x="neighbourhood_group", data=airbnb_new, palette="Greens_d",
              order=airbnb_new.neighbourhood_group.value_counts().index)
fig = plt.gcf()
fig.set_size_inches(8,8)

#Top 10 Neighbourhood
plt.subplot(2, 2, 2)
ax=sns.countplot(x="neighbourhood", data=airbnb_new, palette="Greens_d",
              order=airbnb_new.neighbourhood.value_counts().iloc[:10].index)
fig = plt.gcf()
fig.set_size_inches(8,8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)

#Room type
plt.subplot(2, 2, 3)
sns.countplot(x="room_type", data=airbnb_new, palette="Greens_d",
              order=airbnb_new.room_type.value_counts().index)
fig = plt.gcf()
fig.set_size_inches(8,8)

#Top 10 hosts
plt.subplot(2, 2, 4)
ax=sns.countplot(x="host_id", data=airbnb_new, palette="Greens_d",
              order=airbnb_new.host_id.value_counts().iloc[:10].index)
fig = plt.gcf()
fig.set_size_inches(8,8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
plt.tight_layout()


plt.figure(figsize = (15, 7))
plt.style.use('seaborn-white')
plt.subplot(221)
sns.distplot(airbnb_new['price'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.subplot(222)
sns.distplot(airbnb_new['reviews_per_month'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.subplot(223)
sns.distplot(airbnb_new['number_of_reviews'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.subplot(224)
sns.distplot(airbnb_new['availability_365'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.show()

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 8)
ax = sns.boxplot(x = airbnb_new['neighbourhood_group'], y =airbnb_new['price'], data = airbnb_new, palette = 'Set3')
ax.set_xlabel(xlabel = 'Location', fontsize = 20)
ax.set_ylabel(ylabel = 'Price', fontsize = 20)
ax.set_title(label = 'Distribution of prices acros location', fontsize = 30)
plt.xticks(rotation = 90)

#Code forked from-https://www.kaggle.com/biphili/hospitality-in-era-of-airbnb
plt.style.use('seaborn-white')
f,ax=plt.subplots(1,2,figsize=(18,8))
airbnb_new['room_type'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Share of Room Type')
ax[0].set_ylabel('Room Type Share')
sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = airbnb_new)
ax[1].set_title('Room types occupied by the neighbourhood_group')

f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(airbnb_new.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)


plt.figure(figsize = (15, 15))
plt.style.use('seaborn-white')
plt.subplot(221)
sns.scatterplot(x="latitude", y="longitude",hue="neighbourhood_group", data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(222)
sns.scatterplot(x="latitude", y="longitude",hue="room_type", data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(223)
sns.scatterplot(x="latitude", y="longitude",hue="price", data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(224)
sns.scatterplot(x="latitude", y="longitude",hue="availability_365", data=airbnb_new)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.show()

import pandas as pd
import geopandas as gpd
import math
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
m_1 = folium.Map(location=[40.7128,-74.0060], tiles='cartodbpositron', zoom_start=12)

# Adding a heatmap to the base map
HeatMap(data=airbnb_new[['latitude', 'longitude']], radius=10).add_to(m_1)

# Displaying the map
import webbrowser
output_file = "map.html"
m_1.save(output_file)
webbrowser.open(output_file, new=2)  # open in new tab

number_of_reviews = airbnb_new[(airbnb_new.number_of_reviews.isin(range(50,58)))]
# Creating a map
m_2 = folium.Map(location=[40.7128,-74.0060], tiles='cartodbpositron', zoom_start=13)

# Adding points to the map
for idx, row in number_of_reviews.iterrows():
    Marker([row['latitude'], row['longitude']]).add_to(m_2)

# Displaying the map
output_file = "map2.html"
m_2.save(output_file)
webbrowser.open(output_file, new=2)  # open in new tab

# Creating the map
m_3 = folium.Map(location=[40.7128,-74.0060], tiles='cartodbpositron', zoom_start=13)

# Adding points to the map
mc = MarkerCluster()
for idx, row in number_of_reviews.iterrows():
    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
        mc.add_child(Marker([row['latitude'], row['longitude']]))
m_3.add_child(mc)

# Displaying the map
output_file = "map3.html"
m_3.save(output_file)
webbrowser.open(output_file, new=2)  # open in new tab

# import IPython
# from IPython import display #pretty printing of dataframes in Jupyter notebook
# print("IPython version: {}". format(IPython.__version__))
#
# #Common Model Algorithms
# from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# from xgboost import XGBClassifier
#
# #Common Model Helpers
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn import feature_selection
# from sklearn import model_selection
# from sklearn import metrics