**data_preprocessing.py**

- missing_data = find_missing_values(df)

- df_1 , df_2 = drop_missing_values(df_1, df_2, limit, output_file_path) <br/>
  Creates an excel for missing values in training and / or test data. df can ben None. Default value for limit is 0.15 

- df = labelEnc(df) <br/>
  Creates numerical features from categorical features

- skewed_feats = skewed_features(df) <br/>
  Computes the skewedness of features and return features over 0.75

- df = skewed_features_loglp(df) <br/>
  Computes the skewedness of features and return df with loglp transformed features over 0.75

- column_diff(df1, df2) <br/>
  Print the difference in columns of two dataframes

- describe_report(df, output_file_path=None) <br/>
  Creates a csv report for numerical and categorical features 

- target_correlation(df, feature, k, output_file_path=None) <br/>
  Create a correlation matrix with a target feature. Display the k most correlated features

- create_histograms(df, output_file_path=None)
  Create histograms of all numeric features in df

- round_down(num, divisor) <br/>

- round_up(num, divisor) <br/>

<br/><br/>



**functions_plot.py** 

- outlier_plot(df, feature, target_feature, n_neighbors, output_file_path=None) <br/>
  Plot possible outlier. df = dataframe, feature = possible feature that has outliers, n_neighbors = 10 is good,
  white: no outliers, red: outliers

- plot_learning_curve(estimator, title, X, y, scoring, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), output_file_path=None) <br/>
  Plot the training and cross validated learning curve for an estimator. The scoring has to be matched to the estimator and the problem.

- binary_distplot(df, features, target_feature, output_file_path=None) <br/>
  In a binary classification problem, use this function to see differences in case 0 or 1. If the distribution is similar in both cases,
  the possibility is low, that is feature will predict the classification -> delete this feature. Good is if the distribution is different.
  features = list of features, target_feature: binary target feature

<br/><br/>


**pregression_pipeline.py** 

- pipe_best_params = regression_pipeline(x_train, y_train, pipe, cv, scoring, param_grid) <br/>
  Get the best parameter combination from a pipeline. The scoring has to be matched to the estimator and the problem.

- train_errors = evaluate_pipe_best_train(x_train, y_train, pipe_best, algo, output_file_path=None) <br/>
  (regresion problem) <br/>
  Get the R2 and RMSE for a regression task and the scatter plot with the predicted and target values with a 45 degree red line.