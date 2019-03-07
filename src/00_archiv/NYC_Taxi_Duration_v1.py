#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:39 2017

@author: cdavid
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.grid_search import GridSearchCV


""" 1) Load dataset """
train = pd.DataFrame.from_csv('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/01_data/train.csv')
test = pd.DataFrame.from_csv('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/01_data/test.csv')


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)


def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h # in km


def manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b




def GridSearchCV_plot(results, cv_params, GridSearchCV_plot_x, GridSearchCV_plot_hue):
    scores = [x[1] for x in results]
    scores = np.array(scores).reshape(len(cv_params[GridSearchCV_plot_hue]), len(cv_params[GridSearchCV_plot_x]))
    
    for ind, i in enumerate(cv_params[GridSearchCV_plot_hue]):
        plt.plot(cv_params[GridSearchCV_plot_x], scores[ind], label=GridSearchCV_plot_hue + str(i))
    plt.legend()
    plt.xlabel(GridSearchCV_plot_x)
    plt.ylabel('Mean score')
    plt.show()




""" 2) First look @ the dataset """
#print(train.head())
#print("--------------------------------------------------")
#print(train.info())
#print("--------------------------------------------------")
#print(test.info())
#
#print(train.describe()) # excluding object features
#print(train.describe(include=['O'])) # only object features


""" 3) Check for missing values """
#print(train.isnull().sum())
#print(test.isnull().sum())


""" 4) Check for different measurements and scalings -> features scaling? """
#train.hist(bins=10, figsize=(9,9), grid=False)


""" 5) Ceck for outliers """
#features = list(train)
#for feature in features:
#    try:
#        plt.figure(figsize=(8, 6))
#        plt.title('boxplot '+str(feature))
#        train.boxplot(column=feature)
#        plt.savefig('boxplot'+str(feature)+'.pdf')
#    except:
#        print(str(feature)+' is not numeric')



""" 6) Correlation matrix """
#corr=train.corr()#["Survived"]
#plt.figure(figsize=(10, 10))
#
#sns.heatmap(corr, vmax=.8, linewidths=0.01,
#            square=True,annot=True,cmap='YlGnBu',linecolor="white")
#plt.title('Correlation between features')





""" 7) Feature selection """

# Format to daytime
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')


# get date, weekday, day, month, hour, minute
train['pickup_date'] = train['pickup_datetime'].dt.date
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_day'] = train['pickup_datetime'].dt.day
train['pickup_month'] = train['pickup_datetime'].dt.month
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_minute'] = train['pickup_datetime'].dt.minute

test['pickup_date'] = test['pickup_datetime'].dt.date
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
test['pickup_day'] = test['pickup_datetime'].dt.day
test['pickup_month'] = test['pickup_datetime'].dt.month
test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['pickup_minute'] = test['pickup_datetime'].dt.minute


# calculate haversine distance and manhatten distance
train['distance_haversine'] = haversine_array(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_haversine'] = haversine_array(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)


train['distance_manhattan'] = manhattan_distance(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_manhattan'] = manhattan_distance(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)


# calculate directions north <-> south and east <-> west
train['direction_ns'] = (train.pickup_latitude>train.dropoff_latitude)*1+1
indices = train[(train.pickup_latitude == train.dropoff_latitude) & (train.pickup_latitude!=0)].index
train.loc[indices,'direction_ns'] = 0

test['direction_ns'] = (test.pickup_latitude>test.dropoff_latitude)*1+1
indices = test[(test.pickup_latitude == test.dropoff_latitude) & (test.pickup_latitude!=0)].index
test.loc[indices,'direction_ns'] = 0


train['direction_ew'] = (train.pickup_longitude>train.dropoff_longitude)*1+1
indices = train[(train.pickup_longitude == train.dropoff_longitude) & (train.pickup_longitude!=0)].index
train.loc[indices,'direction_ew'] = 0

test['direction_ew'] = (test.pickup_longitude>test.dropoff_longitude)*1+1
indices = test[(test.pickup_longitude == test.dropoff_longitude) & (test.pickup_longitude!=0)].index
test.loc[indices,'direction_ew'] = 0



# create map with trips
#fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
#ax[0].scatter(train['pickup_longitude'].values, train['pickup_latitude'].values,
#              color='blue', s=1, label='train', alpha=0.1)
#ax[1].scatter(test['pickup_longitude'].values, test['pickup_latitude'].values,
#              color='green', s=1, label='test', alpha=0.1)
#fig.suptitle('Train and test area complete overlap.')
#ax[0].legend(loc=0)
#ax[0].set_ylabel('latitude')
#ax[0].set_xlabel('longitude')
#ax[1].set_xlabel('longitude')
#ax[1].legend(loc=0)
#plt.ylim(city_lat_border)
#plt.xlim(city_long_border)
#plt.show()




# create average speed in training dataset -> average speed in NYC district is the same of test dataset

train['avg_speed_h'] = 3600 * train['distance_haversine'] / train['trip_duration']
train['avg_speed_m'] = 3600 * train['distance_manhattan'] / train['trip_duration']

#plt.plot(train.groupby('pickup_hour').mean()['avg_speed_h'])
#plt.plot(train.groupby('pickup_hour').mean()['avg_speed_m'])
#plt.xlabel('hour')
#plt.ylabel('average speed')



from sklearn.cluster import MiniBatchKMeans

train['pickup_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(train[['pickup_latitude', 'pickup_longitude']])
train['dropoff_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(train[['dropoff_latitude', 'dropoff_longitude']])

test['pickup_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(test[['pickup_latitude', 'pickup_longitude']])
test['dropoff_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(test[['dropoff_latitude', 'dropoff_longitude']])


#cm = plt.cm.get_cmap('RdYlBu')
#fig, ax = plt.subplots(ncols=1, nrows=1)
#ax.scatter(train.pickup_longitude.values, train.pickup_latitude.values,c=train.pickup_cluster.values,
#           alpha=0.2, s=10, cmap=cm)
#ax.set_xlim(city_long_border)
#ax.set_ylim(city_lat_border)
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')
#plt.show()










##################################################################################
""" Detailed analysis of drip_duration """

# remove outliers in trip_duration
q = train['trip_duration'].quantile(0.99) # remove the highest 1%
train = train[train.trip_duration < q]
#train.boxplot(column='trip_duration')

# correced distribution
#sns.distplot(train['trip_duration'], bins=50, kde=True) # right skewed distribution ->apply logarithm
#sns.distplot(np.log(train['trip_duration']), bins=50, kde=True)
train['trip_duration'] = np.log(train['trip_duration'])


##################################################################################
""" Detailed analysis of number of trips """

#train['pickup_weekday_'] = train['pickup_weekday'].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
#     4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
#
#train['pickup_weekday_weekend'] = train['pickup_weekday'].replace({0: 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday',
#     4: 'weekday', 5: 'weekend', 6: 'weekend'})
#    
#plt.figure(figsize=(15, 5))
#sns.countplot(x='pickup_hour', hue='pickup_weekday_', data=train, hue_order=['Monday','Tuesday','Wednesday', 'Thursday',
#                                                                        'Friday', 'Saturday', 'Sunday'])
#plt.savefig('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour-pickup_weekday.pdf')
#
#plt.figure(figsize=(15, 5))
#sns.countplot(x='pickup_hour', hue='pickup_weekday_weekend', data=train, hue_order=['weekday','weekend'])
#plt.savefig('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour-pickup_weekday_weekend.pdf')
#
#
#a = train.groupby(['pickup_hour', 'pickup_weekday_weekend']).count()
#a = a['vendor_id']
#a = a.reset_index()
#a['vendor_id'][a['pickup_weekday_weekend'] == 'weekday'] = a['vendor_id'][a['pickup_weekday_weekend'] == 'weekday']/5
#a['vendor_id'][a['pickup_weekday_weekend'] == 'weekend'] = a['vendor_id'][a['pickup_weekday_weekend'] == 'weekend']/2
#a = a.groupby(['pickup_hour', 'pickup_weekday_weekend']).sum()
#a = a.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
#a = a.reset_index()
#
#sns.axes_style("darkgrid")
#sns.factorplot(x="pickup_hour", y="vendor_id", hue="pickup_weekday_weekend", data=a, legend=None, size=4, aspect=2)
#plt.ylabel('pickup distribution')
#plt.legend(loc='upper center')
#plt.grid(color='b', linestyle='-', linewidth=1)
#plt.savefig('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour_distribution.pdf')


##################################################################################
""" Detailed analysis of avg_speed """

train = train[train['avg_speed_h'] < 150]
train = train[train['avg_speed_m'] < 150]



##################################################################################
""" Detailed analysis of latitude % longitude """

train = train[train['pickup_latitude'] > 40.63]
train = train[train['pickup_longitude'] < 40.85]

train = train[train['dropoff_latitude'] > -74.03]
train = train[train['dropoff_longitude'] < -73.75]








##################################################################################


dropp_diff = list(np.setdiff1d(train.columns, test.columns))
do_not_use_for_training = ['vendor_id', 'pickup_datetime', 'pickup_date']

cols_to_drop = dropp_diff + do_not_use_for_training
features = [f for f in train.columns if f not in cols_to_drop]


#train = train.ix[:50000,:]
#test = test.ix[:3000,:]

y_train = train['trip_duration']
X_train = train[features]
X_test = test[features]




#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

xgdmat = xgb.DMatrix(X_train, y_train)




##########################
# Grid search
cv_params = {
    'max_depth': [5,12],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 1000, 5000],
    'min_child_weight': [4],
    'gamma':[0.3],
    'subsample':[0.9],
    'colsample_bytree':[0.9],
    'reg_alpha':[4]
    
    }

xclas = xgb.XGBRegressor(nthread=-1) 
gs = GridSearchCV(xclas, cv_params, n_jobs=1) 
grid_result = gs.fit(X_train, y_train)

print('Best score: '+ str(gs.best_score_))
print('Best parameters: ' + str(gs.best_params_))


#results = gs.grid_scores_
#GridSearchCV_plot_x = 'learning_rate'
#GridSearchCV_plot_hue = 'n_estimators'
#GridSearchCV_plot(results, cv_params, GridSearchCV_plot_x, GridSearchCV_plot_hue)


##########################



best_params = gs.best_params_

cv_xgb = xgb.cv(params = best_params, dtrain = xgdmat, nfold = 5,
                metrics = ['rmse'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error


final_gb = xgb.train(best_params, xgdmat, num_boost_round = 200)
#xgb.plot_importance(final_gb)


""" Testing """
from sklearn.metrics import r2_score

testdmat = xgb.DMatrix(X_test)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat



#print('R^2: ' + str(r2_score(y_pred, y_test)))





""" create submission """
id_test = test['vendor_id'].values
df = pd.DataFrame({'id': X_test.index, 'trip_duration': y_pred}) 
df = df.set_index('id')
df.to_csv('/home/cdavid/Documents/01_Programmieren/Python Projects/NYC_Taxi_Duration/01_data/submission.csv', index = True)

print('Finished')


