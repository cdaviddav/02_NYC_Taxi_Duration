import matplotlib.pyplot as plt
import os
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import seaborn as sns


def example_plot(data, output_file_path=None):
    fig, ax = plt.subplots()
    data.hist(ax=ax)

    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'example_figure.pdf'))



def outlier_plot(df, feature, target_feature, n_neighbors, output_file_path=None):

    if n_neighbors is None:
        n_neighbors = 20

    clf = LocalOutlierFactor(n_neighbors, n_jobs=-1)
    y_pred = clf.fit_predict(df[[feature, target_feature]])

    get_outliers = lambda y_pred, xs: [i for (y, i) in zip(xs, range(len(xs))) if y_pred == y]
    outliers_true = get_outliers(-1,y_pred)
    outliers_false = get_outliers(1,y_pred)

    df_train_outliers_true = df[[feature, target_feature]].iloc[outliers_true]
    df_train_outliers_false = df[[feature, target_feature]].iloc[outliers_false]

    # plot the level sets of the decision function
    fig, ax = plt.subplots()
    plt.scatter(df_train_outliers_true[feature], df_train_outliers_true[target_feature], c='red', edgecolor='k', s=20, label='outliers_true')
    plt.scatter(df_train_outliers_false[feature], df_train_outliers_false[target_feature], c='white', edgecolor='k', s=20, label='outliers_false')
    ax.legend()

    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'outlier_' + str(feature) + '.pdf'))


def plot_learning_curve(estimator, title, X, y, scoring, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), output_file_path=None):

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

    if output_file_path is not None:
        plt.savefig(os.path.join(output_file_path, 'learning_curve_' + str(title) + '.pdf'))


def binary_distplot(df, features, target_feature, output_file_path=None):
    for v_feature in features:
        fig, ax = plt.subplots()
        sns.distplot(df[v_feature][df.target_feature == 1], bins=50, color='r')
        sns.distplot(df[v_feature][df.target_feature == 0], bins=50, color='b')
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(v_feature))
        if output_file_path is not None:
            fig.savefig(os.path.join(output_file_path, 'distplot_' + str(v_feature) + '.pdf'))


def plot_validation_parameter_curve(estimator, title, X, y, log_x_axes, param_name, param_range, scoring):

    plt.figure()
    plt.title(title)
    plt.xlabel("n_neighbors")
    plt.ylabel("Score")

    param_range = np.linspace(1, 50, num=50).astype(int)
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=5, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    lw = 2

    if log_x_axes == True:
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

    else:
        plt.plot(train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.plot(test_scores_mean, label="Cross-validation score", color="navy", lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt


def create_histograms(df, output_file_path=None):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_features = list(df.select_dtypes(include=numerics))

    for num_feature in num_features:
        fig, ax = plt.subplots()
        df[num_feature].hist(bins=50, ax=ax)
        plt.title(str(num_feature))
        fig.savefig(os.path.join(output_file_path, 'hist_' + str(num_feature) + '.pdf'))


def create_boxplots(df, output_file_path=None):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_features = list(df.select_dtypes(include=numerics))

    for num_feature in num_features:
        fig, ax = plt.subplots()
        df.boxplot(column=num_feature, ax=ax)
        plt.title(str(num_feature))
        fig.savefig(os.path.join(output_file_path, 'box_' + str(num_feature) + '.pdf'))