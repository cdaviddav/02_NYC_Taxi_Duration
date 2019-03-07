import os
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def find_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def drop_missing_values(df_1, df_2, limit, output_file_path):
    # limit: How much data has to be missing to delete column

    if limit is None:
        limit = 0.15

    if df_1 is not None:
        missing_data_1 = find_missing_values(df_1)

        df_1 = df_1.drop(missing_data_1[missing_data_1['Percent'] > limit].index, axis=1)

        delete_row_cols_df_1 = missing_data_1[
            (missing_data_1['Percent'] <= limit) & (missing_data_1['Percent'] > 0)].index
        for row in delete_row_cols_df_1:
            df_1 = df_1.drop(df_1.loc[df_1[row].isnull()].index)

        # control if there are no missing values left
        print("Missing values left: " + str(df_1.isnull().sum().sum()))

    else:
        missing_data_1 = None


    if df_2 is not None:
        missing_data_2 = find_missing_values(df_2)

        df_2 = df_2.drop(missing_data_2[missing_data_2['Percent'] > limit].index, axis=1)

        delete_row_cols_df_1 = missing_data_2[
            (missing_data_2['Percent'] <= limit) & (missing_data_2['Percent'] > 0)].index
        for row in delete_row_cols_df_1:
            df_2 = df_2.drop(df_2.loc[df_2[row].isnull()].index)

        # control if there are no missing values left
        print("Missing values left: " + str(df_2.isnull().sum().sum()))

    else:
        missing_data_2 = None


    if output_file_path is not None:
        writer = pd.ExcelWriter(os.path.join(output_file_path, 'missing_values.xlsx'))
        if missing_data_1 is not None:
            missing_data_1.to_excel(writer, 'train')
        if missing_data_2 is not None:
            missing_data_2.to_excel(writer, 'validate')
        writer.save()

    return df_1, df_2


def labelEnc(df):
    labelEnc = LabelEncoder()
    cat_vars = list(df.describe(include=['O']).columns)
    for col in cat_vars:
        try:
            df[col] = labelEnc.fit_transform(df[col])
        except:
            print('LabelEncoder Error: ' + str(col))
    return df


def skewed_features(df):
    #log transform skewed numeric features:
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    return skewed_feats


def skewed_features_loglp(df):
    #log transform skewed numeric features:
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    df[skewed_feats] = np.log1p(df[skewed_feats])
    return df


def column_diff(df1, df2):
    print("Columns in " + str(df1.name) + " not in " + str(df2.name))
    print(df1.columns.difference(df2.columns))
    print("##################")
    print("Columns in " + str(df2.name) + " not in " + str(df1.name))
    print(df2.columns.difference(df1.columns))


def describe_report(df, output_file_path=None):
    if output_file_path is not None:
        df.describe().transpose().to_csv(os.path.join(output_file_path, str(df.name) + '_describe_numeric.csv'))
        df.describe(include=['O']).transpose().to_csv(os.path.join(output_file_path, str(df.name) + '_describe_categorical.csv'))


def target_correlation(df, feature, k, output_file_path=None):
    cols = df.corr().nlargest(k, feature)[feature].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.00)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
    if output_file_path is not None:
        plt.savefig(os.path.join(output_file_path, 'correlation_matrix.pdf'))


