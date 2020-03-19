# kaggle-house-price-prediction-regression-in-top28-leaderboard

##**House Price Prediction** 

This is a very interesting competition provided by Kaggle. I believe that all the entry level Machine Learning enthusiasts should definitely get their hands-on to this competition as well as the [Titanic Survivor Prediction competition ](https://www.kaggle.com/c/titanic). One can also check [my notebook](https://www.kaggle.com/darshanjain29/titanic-survival-from-top-70-to-top-7-on-lb) on the same.

Click to see my [kaggle solution](https://www.kaggle.com/darshanjain29/house-price-prediction-top-in-94-to-top-in-28/)

Coming back to this competition, as it says that the problem is predicting house price i.e. a continuous value. Hence it is a regression problem and thus all your basic regression knowledge will be tested here.

Now, from a beginner's perspective how do you start with this? Well, I will show very easy steps with which you can easily jump from **top 95%** on leaderboard to **top 28%**. So, let's get started.

1. Let's read about the data that is given. So, in the train.csv there are 1460 rows and 81 columns whereas in test.csv there are 1459 and 80 columns.
2. After reading the file, we start data preprocessing and feature engineering. Before moving ahead we are combining both the dataframes so that changes can be done in both the dfs together.
3. So, with the help of heatmap we can check number of nulls in each feature. So, based on the column name and after reading the description in the file data_description.txt of that column, we can decide if we want to replace nulls by mean(), mode(), median() or 'NA' or something else.
4. Now, we have to replace all the object/string values to numerical type using one hot encoding. While doing this many new columns will be created. So, we also run a code to drop all duplicate columns if any
5. Now I am separating the train and test data from all_data and appending SalePrice column to the train data and the training models starts from here.
6. While training the model, we are using Kfold cross validation for the better root mean squared log error as it is the evaluation criteria on the leaderboard.
7. Training with Simple linear regression model : r2_score that we got was a negative number thus we were getting error with rmsle as it was not able to handle negative number
8. Training with Lasso regression : r2_score was a positive number but rmsle was throwing error because of negative values in the data
9. Training with ridge model: rmsle, r2_score = 0.1562332068843027, 0.8350220277268043 respectively
10. Training with Elastic Net: rmsle, r2_score = 0.15524191588871572, 0.8355319656725282 respectively
11. Training with SVM algorithm: rmsle, r2_score = 0.42655965911720095, -0.21383191346231967 respectively
12. Training with Decision Tree: rmsle, r2_score = 0.20087638073362202, 0.7569636210379455 respectively
13. Training with Random Forest Regressor model: rmsle, r2_score = 0.17295960323425913, 0.8220478028104417 respectively
14. Training with Adaboost regressor model: rmsle, r2_score = 0.20880462846912984, 0.7886447451714611 respectively
15. Training with Bagging Classifier model: rmsle, r2_score = 0.14719806924931875, 0.8539938038399392 respectively
16. Training with XGboost model: rmsle, r2_score = 0.13372606548796895, 0.8742884283414025 respectively
17. Training with Gradient Boosting Regressor model: rmsle, r2_score = 0.14187482285610134, 0.8865067659268699 respectively
18. Now you can do a submission and check your score on the leaderboard.
19. After trying each model, try using the same one with hyper parameter tuning and you will be able to small improvement in rmsle and r2_score as well as your position on the leaderboard
20. At the end we have check the feature importance count and removed all the features with importance count less than 0. 

I hope this helps. Please comment below if you haven't understood anything from the above steps.
