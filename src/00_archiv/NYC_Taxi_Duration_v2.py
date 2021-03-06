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
#import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import statsmodels.api as sm


""" 1) Load dataset """
df_train = pd.DataFrame.from_csv(r"D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/01_data/train.csv")
df_test = pd.DataFrame.from_csv(r"D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/01_data/test.csv")


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)



def create_submission(x_test, y_val_predict):
    """ create submission """
    id_test = test['vendor_id'].values
    df = pd.DataFrame({'id': x_test.index, 'trip_duration': y_val_predict}) 
    df = df.set_index('id')
    df.to_csv(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/01_data/submission.csv', index = True)


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


def find_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def drop_missing_values(df_train, df_test):
    missing_data_train = find_missing_values(df_train)
    missing_data_test = find_missing_values(df_test)

    df_train = df_train.drop(missing_data_train[missing_data_train['Percent'] > 0.15].index, axis=1)
    df_test = df_test.drop(missing_data_test[missing_data_test['Percent'] > 0.15].index, axis=1)

    delete_row_cols_train = missing_data_train[
        (missing_data_train['Percent'] <= 0.15) & (missing_data_train['Percent'] > 0)].index
    for row in delete_row_cols_train:
        df_train = df_train.drop(df_train.loc[df_train[row].isnull()].index)

    delete_row_cols_test = missing_data_test[
        (missing_data_test['Percent'] <= 0.15) & (missing_data_test['Percent'] > 0)].index
    for row in delete_row_cols_test:
        df_test = df_test.drop(df_test.loc[df_test[row].isnull()].index)

    # control if there are no missing values left
    print(df_train.isnull().sum().sum())
    print(df_test.isnull().sum().sum())
    return df_train, df_test


def check_dataset(train, test):
    """ 2) First look @ the dataset """
    #print(df_train.head())
    #print("--------------------------------------------------")
    #print(df_train.info())
    #print("--------------------------------------------------")
    #print(test.info())
    #
    #print(df_train.describe()) # excluding object features
    #print(df_train.describe(include=['O'])) # only object features
    
    
    """ 3) Check for missing values """    
    find_missing_values(df_train)
    find_missing_values(test)
    
    
    """ 4) Check for different measurements and scalings -> features scaling? """
    #df_train.hist(bins=10, figsize=(9,9), grid=False)
    
    
    """ 5) Ceck for outliers """
    features = list(df_train)
    for feature in features:
        try:
            plt.figure(figsize=(8, 8))
            plt.title('boxplot '+str(feature))
            df_train.boxplot(column=feature)
            plt.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/boxplot_'+str(feature)+'.pdf')
        except:
            print(str(feature)+' is not numeric')
    
    
    
    """ 6) Correlation matrix """
    corr=df_train.corr()#["Survived"]
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(corr, vmax=.8, linewidths=0.01,
                square=True,annot=True,cmap='YlGnBu',linecolor="white")
    plt.title('Correlation between features')
    plt.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/correlation_matrix.pdf')



    # create map with trips
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].scatter(df_train['pickup_longitude'].values, df_train['pickup_latitude'].values,
                  color='blue', s=1, label='train', alpha=0.1)
    ax[1].scatter(test['pickup_longitude'].values, test['pickup_latitude'].values,
                  color='green', s=1, label='test', alpha=0.1)
    fig.suptitle('Train and test area complete overlap.')
    ax[0].legend(loc=0)
    ax[0].set_ylabel('latitude')
    ax[0].set_xlabel('longitude')
    ax[1].set_xlabel('longitude')
    ax[1].legend(loc=0)
    plt.ylim(city_lat_border)
    plt.xlim(city_long_border)
    plt.show()


def clean_outliers(df_train, feature_list):
    liste = list()
    df_train['outlier_test_remember'] = 'no_outlier'
    for feature in feature_list:
        df_train['outlier_test'] = 'no_outlier'
        try:
            x = df_train[feature]
            y = df_train['SalePrice']
            model = sm.OLS(y, x)
            results = model.fit()

            test = results.outlier_test()
            test2 = test[test['bonf(p)'] < 0.5]
            liste.append(list(test2.index))
            print('There are ' + str(test2.shape[0]) + ' Outliers -> ' + str(
                round(test2.shape[0] / df_train.shape[0] * 100, 2)) + '%')


            for index in list(test2.index):
                df_train = df_train.set_value(index, 'outlier_test', 'outlier')
                df_train = df_train.set_value(index, 'outlier_test_remember', 'outlier')

            outlier_plot = sns.lmplot(x=str(feature), y="SalePrice", hue="outlier_test", data=df_train, fit_reg=False)
            outlier_plot.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/outlier_test/outlier_' + str(feature) + '.pdf')
        except: pass

    liste = [item for sublist in liste for item in sublist]  # flatten list
    liste = list(set(liste))  # drop dupplicates in list

    return df_train, liste


def plot_learning_curve(estimator, title, X, y, scoring, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/' + str(title) + '.pdf')


def plot_average_speed(train):
    plt.plot(train.groupby('pickup_hour').mean()['avg_speed_h'])
    plt.plot(train.groupby('pickup_hour').mean()['avg_speed_m'])
    plt.xlabel('hour')
    plt.ylabel('average speed')
    plt.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/average_speed_perHour.pdf')


def plot_cluster_map(df_train, df_test):
    from sklearn.cluster import MiniBatchKMeans
    # cluster trips into MiniBatches and plot cluster on map

    df_train['pickup_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(
        df_train[['pickup_latitude', 'pickup_longitude']])
    df_train['dropoff_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(
        df_train[['dropoff_latitude', 'dropoff_longitude']])

    df_test['pickup_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(
        df_test[['pickup_latitude', 'pickup_longitude']])
    df_test['dropoff_cluster'] = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit_predict(
        df_test[['dropoff_latitude', 'dropoff_longitude']])

    cm = plt.cm.get_cmap('RdYlBu')
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.scatter(df_train.pickup_longitude.values, df_train.pickup_latitude.values, c=df_train.pickup_cluster.values,
               alpha=0.2, s=10, cmap=cm)
    ax.set_xlim(city_long_border)
    ax.set_ylim(city_lat_border)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()


def plot_pickup_hour_pickup_weekday(train):
    plt.figure(figsize=(15, 5))
    sns.countplot(x='pickup_hour', hue='pickup_weekday_', data=train,
                  hue_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                             'Friday', 'Saturday', 'Sunday'])
    plt.savefig(
        r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour-pickup_weekday.pdf')

    plt.figure(figsize=(15, 5))
    sns.countplot(x='pickup_hour', hue='pickup_weekday_weekend', data=train, hue_order=['weekday', 'weekend'])
    plt.savefig(
        r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour-pickup_weekday_weekend.pdf')


def plot_hour_distribution(df):
    a = df_train.groupby(['pickup_hour', 'pickup_weekday_weekend']).count()
    a = a['vendor_id']
    a = a.reset_index()
    a['vendor_id'][a['pickup_weekday_weekend'] == 'weekday'] = a['vendor_id'][a['pickup_weekday_weekend'] == 'weekday'] / 5
    a['vendor_id'][a['pickup_weekday_weekend'] == 'weekend'] = a['vendor_id'][a['pickup_weekday_weekend'] == 'weekend'] / 2
    a = a.groupby(['pickup_hour', 'pickup_weekday_weekend']).sum()
    a = a.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    a = a.reset_index()

    sns.axes_style("darkgrid")
    sns.factorplot(x="pickup_hour", y="vendor_id", hue="pickup_weekday_weekend", data=a, legend=None, size=4, aspect=2)
    plt.ylabel('pickup distribution')
    plt.legend(loc='upper center')
    plt.grid(color='b', linestyle='-', linewidth=1)
    plt.savefig(r'D:/01_Programmieren/1_Machine_Learning/Python Projects/NYC_Taxi_Duration/03_plots/pickup_hour_distribution.pdf')




df_train, df_test = drop_missing_values(df_train, df_test)
#check_dataset(df_train, df_test)

feature_list = list(df_train.columns)
feature_list.remove('trip_duration')

df_train, list_outliers = clean_outliers(df_train, feature_list)
df_train = df_train[df_train['outlier_test_remember'] == 'no_outlier']



""" 7) Feature selection """

# Format to daytime
df_train['pickup_datetime'] = pd.to_datetime(df_train.pickup_datetime)
df_train['dropoff_datetime'] = pd.to_datetime(df_train.dropoff_datetime)
df_train['store_and_fwd_flag'] = 1 * (df_train.store_and_fwd_flag.values == 'Y')

df_test['pickup_datetime'] = pd.to_datetime(df_test.pickup_datetime)
df_test['store_and_fwd_flag'] = 1 * (df_test.store_and_fwd_flag.values == 'Y')


# get date, weekday, day, month, hour, minute
df_train['pickup_date'] = df_train['pickup_datetime'].dt.date
df_train['pickup_weekday'] = df_train['pickup_datetime'].dt.weekday
df_train['pickup_day'] = df_train['pickup_datetime'].dt.day
df_train['pickup_month'] = df_train['pickup_datetime'].dt.month
df_train['pickup_hour'] = df_train['pickup_datetime'].dt.hour
df_train['pickup_minute'] = df_train['pickup_datetime'].dt.minute

df_test['pickup_date'] = df_test['pickup_datetime'].dt.date
df_test['pickup_weekday'] = df_test['pickup_datetime'].dt.weekday
df_test['pickup_day'] = df_test['pickup_datetime'].dt.day
df_test['pickup_month'] = df_test['pickup_datetime'].dt.month
df_test['pickup_hour'] = df_test['pickup_datetime'].dt.hour
df_test['pickup_minute'] = df_test['pickup_datetime'].dt.minute


# calculate haversine distance and manhatten distance
df_train['distance_haversine'] = haversine_array(
        df_train['pickup_latitude'].values, df_train['pickup_longitude'].values,
        df_train['dropoff_latitude'].values, df_train['dropoff_longitude'].values)

df_test['distance_haversine'] = haversine_array(
    df_test['pickup_latitude'].values, df_test['pickup_longitude'].values,
    df_test['dropoff_latitude'].values, df_test['dropoff_longitude'].values)


df_train['distance_manhattan'] = manhattan_distance(
        df_train['pickup_latitude'].values, df_train['pickup_longitude'].values,
        df_train['dropoff_latitude'].values, df_train['dropoff_longitude'].values)

df_test['distance_manhattan'] = manhattan_distance(
    df_test['pickup_latitude'].values, df_test['pickup_longitude'].values,
    df_test['dropoff_latitude'].values, df_test['dropoff_longitude'].values)


# calculate directions north <-> south (latutude) and east <-> west (longitude)
# +1 because TRUE and FALSE is represented by 0 and 1 -> +1 to shift representation to 1 and 2 because same latitude represented by 0
df_train['direction_ns'] = (df_train.pickup_latitude>df_train.dropoff_latitude)*1+1
indices = df_train[(df_train.pickup_latitude == df_train.dropoff_latitude) & (df_train.pickup_latitude!=0)].index
df_train.loc[indices,'direction_ns'] = 0

df_test['direction_ns'] = (df_test.pickup_latitude>df_test.dropoff_latitude)*1+1
indices = df_test[(df_test.pickup_latitude == df_test.dropoff_latitude) & (df_test.pickup_latitude!=0)].index
df_test.loc[indices,'direction_ns'] = 0

df_train['direction_ew'] = (df_train.pickup_longitude>df_train.dropoff_longitude)*1+1
indices = df_train[(df_train.pickup_longitude == df_train.dropoff_longitude) & (df_train.pickup_longitude!=0)].index
df_train.loc[indices,'direction_ew'] = 0

df_test['direction_ew'] = (df_test.pickup_longitude>df_test.dropoff_longitude)*1+1
indices = df_test[(df_test.pickup_longitude == df_test.dropoff_longitude) & (df_test.pickup_longitude!=0)].index
df_test.loc[indices,'direction_ew'] = 0


# create average speed in training dataset -> average speed in NYC district is the same of test dataset
df_train['avg_speed_h'] = 3600 * df_train['distance_haversine'] / df_train['trip_duration']
df_train['avg_speed_m'] = 3600 * df_train['distance_manhattan'] / df_train['trip_duration']


#plot_average_speed(df_train)
#plot_cluster_map(df_train, df_test)




##################################################################################
""" Detailed analysis of drip_duration """

# remove outliers in trip_duration
q = df_train['trip_duration'].quantile(0.99) # remove the highest 1%
df_train = df_train[df_train.trip_duration < q]

duration = pd.DataFrame({"duration":df_train["trip_duration"], "loglp(duration + 1)":np.log1p(df_train["trip_duration"]), "log(duration + 1)":np.log(df_train["trip_duration"])})
duration.hist(bins=50)
df_train["trip_duration"] = np.log1p(df_train["trip_duration"])


##################################################################################
""" Detailed analysis of number of trips """

df_train['pickup_weekday_'] = df_train['pickup_weekday'].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
     4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

df_train['pickup_weekday_weekend'] = df_train['pickup_weekday'].replace({0: 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday',
     4: 'weekday', 5: 'weekend', 6: 'weekend'})

#plot_pickup_hour_pickup_weekday(df_train)
#plot_hour_distribution(df_train)


##################################################################################
""" Detailed analysis of avg_speed """

df_train = df_train[df_train['avg_speed_h'] < 150]
df_train = df_train[df_train['avg_speed_m'] < 150]



##################################################################################
""" Detailed analysis of latitude % longitude """

df_train = df_train[df_train['pickup_latitude'] > 40.63]
df_train = df_train[df_train['pickup_longitude'] < 40.85]

df_train = df_train[df_train['dropoff_latitude'] > -74.03]
df_train = df_train[df_train['dropoff_longitude'] < -73.75]



##################################################################################

dropp_diff = list(np.setdiff1d(df_train.columns, df_test.columns))
do_not_use_for_training = ['vendor_id', 'pickup_datetime', 'pickup_date']

cols_to_drop = dropp_diff + do_not_use_for_training
features = [f for f in df_train.columns if f not in cols_to_drop]

y_train = df_train['trip_duration']
df_train = df_train[list(df_test)]
df_train.drop(['vendor_id', 'pickup_datetime', 'pickup_date'], axis=1, inplace=True)
df_test.drop(['vendor_id', 'pickup_datetime', 'pickup_date'], axis=1, inplace=True)


df_train = df_train.ix[:50000,:]
y_train = y_train[:50000]
df_test = df_test.ix[:3000,:]

x_train = df_train[features]
x_test = df_test[features]







from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# create pipeline with machine learning process
pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('feature_scaling', MinMaxScaler()), # scaling because linear models are sensitive to the scale of input features
        ('regression', Ridge()), # ridge is default algorithm for linear models
        ])

# create parameter grid for hyperparamter tuning with grid search
param_grid = [{'reduce_dim__n_components': [5, 10, 15],
               'regression__alpha': [0.005, 0.05, 0.5, 0.1],
               'regression__solver': ['svd', 'cholesky']
               }]

# create model with grid search and fill the pipeline and param_grid (testing grid search though cross validation)
reg = GridSearchCV(pipe, cv=5, scoring='r2', param_grid=param_grid)
reg.fit(x_train, y_train)

# get the explained variance achieved with dimension reduction -> see if # of dimensions can be reduced further
#reg.best_estimator_.named_steps['reduce_dim'].explained_variance_


# get scores for regression model
means = reg.cv_results_['mean_test_score']
stds = reg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, reg.cv_results_['params']):
    print(str(reg.scoring) + " Score: " +  "%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))




# get best parameter to fit them into the testing model
pipe_best_params = reg.best_params_
pipe_best = Pipeline([
        ('reduce_dim', PCA(n_components = pipe_best_params['reduce_dim__n_components'])),
        ('feature_scaling', MinMaxScaler()),
        ('regression', Ridge(alpha = pipe_best_params['regression__alpha'],solver = pipe_best_params['regression__solver'])),
        ])



from sklearn.metrics import mean_squared_error
# train the testing model and print the training and testing/validation error -> see if bias or variance problem
train_errors, val_errors = [], []
for m in range(1, len(x_train), 100):
    pipe_best.fit(x_train[:m], y_train[:m])
    y_train_predict = pipe_best.predict(x_train[:m])
    # y_train_predict = np.expm1(y_train_predict)
    train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

print('R2: {}'.format(r2_score(y_train_predict, np.expm1(y_train[:len(y_train_predict)]))))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_train_predict, np.expm1(y_train[:len(y_train_predict)])))))

fig = plt.figure()
plt.scatter(y_train_predict[:49000], y_train[:49000])
plt.plot([min(y_train_predict),max(y_train_predict)], [min(y_train_predict),max(y_train_predict)], c="red")
fig.savefig('validation.pdf')


plot_learning_curve(pipe_best, 'ridge_regression_learningCurve', x_train, y_train, 'r2')



#create_submission(x_test, y_val_predict)

