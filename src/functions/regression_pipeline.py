import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os

def regression_pipeline(x_train, y_train, pipe, cv, scoring, param_grid):

    cv_ = cv
    scoring_ = scoring
    param_grid_ = param_grid

    reg = GridSearchCV(pipe, cv = cv_, scoring=scoring_, param_grid=param_grid_)
    reg.fit(x_train, y_train)

    # get the explained variance achieved with dimension reduction -> see if # of dimensions can be reduced further
    #reg.best_estimator_.named_steps['reduce_dim'].explained_variance_


    # get scores for regression model
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print(str(reg.scoring) + " Score: " +  "%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print('Best ' + str(reg.scoring) + " Score: " + str(max(means)))
    pipe_best_params = reg.best_params_

    # get best parameter to fit them into the testing model
    return pipe_best_params


def evaluate_pipe_best_train(x_train, y_train, pipe_best, algo, log, output_file_path=None):

    # train the testing model and print the training and testing/validation error -> see if bias or variance problem
    train_errors = []
    len_x_train = len(x_train) - (len(x_train)%10)
    for m in range(1, len_x_train, 10):
        pipe_best.fit(x_train[:m], y_train[:m])
        y_train_predict = pipe_best.predict(x_train[:m])
        #print(x_train[:m])
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

    if log == True:
        y_train_predict = np.expm1(y_train_predict)
        y_train = np.expm1(y_train)
    else:
        y_train_predict = y_train_predict
        y_train = y_train

    print('R2: {}'.format(r2_score(y_pred = y_train_predict[:min(len(y_train_predict), len_x_train)],
                                   y_true = y_train[:min(len(y_train_predict), len_x_train)])))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_train_predict[:min(len(y_train_predict), len_x_train)],
                                                       y_train[:min(len(y_train_predict), len_x_train)]))))

    fig = plt.figure()
    plt.scatter(y_train_predict[:min(len(y_train_predict), len_x_train)],
                y_train[:min(len(y_train_predict), len_x_train)])
    plt.plot([min(y_train_predict),max(y_train_predict)], [min(y_train_predict),max(y_train_predict)], c="red")
    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'evaluate_pipe_best_train_' + algo +'.pdf'))

    return train_errors


def evaluate_pipe_best_test(x_test, y_test, pipe_best, algo, log, output_file_path=None):
    len_x_test = len(x_test) - (len(x_test) % 10)
    y_test_predict = pipe_best.predict(x_test)

    if log == True:
        y_test_predict = np.expm1(y_test_predict)
        y_test = np.expm1(y_test)
    else:
        y_test_predict = y_test_predict
        y_test = y_test

    print('R2: {}'.format(r2_score(y_pred = y_test_predict[:min(len(y_test_predict), len_x_test)],
                                   y_true = y_test[:min(len(y_test_predict), len_x_test)])))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_test_predict[:min(len(y_test_predict), len_x_test)],
                                                       y_test[:min(len(y_test_predict), len_x_test)]))))

    fig = plt.figure()
    plt.scatter(y_test_predict[:min(len(y_test_predict), len_x_test)],
                y_test[:min(len(y_test_predict), len_x_test)])
    plt.plot([min(y_test_predict),max(y_test_predict)], [min(y_test_predict),max(y_test_predict)], c="red")
    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'evaluate_pipe_best_test_' + algo +'.pdf'))