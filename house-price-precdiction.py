# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Plotting started code requied
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
import seaborn as sns

train_df  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_df.info()
train_df.describe()
#train_df.head(10) 
##: 1460 rows and 81 columns

test_df  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_df.info()
test_df.describe()
#train_df.head(10)
##: 1459 rows and 81 columns

#To check the nulls on a heatmap
sns.heatmap(train_df.isnull(), yticklabels = False, cbar = False)

#Combining both the dataframes so that changes can be done in both the dfs together
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
x_saleprice = train_df["SalePrice"]
all_data.drop(["SalePrice"], axis = 1, inplace= True)
all_data.shape
##(2919, 80)

#To check null with column name
x = all_data.isnull().sum().sort_values(ascending = False)
print (x[x>0])

#Data preprocessing and feature engineering
all_data["PoolQC"] = all_data["PoolQC"].fillna("NA")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("NA")
all_data["Alley"] = all_data["Alley"].fillna("NA")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("NA")
all_data["Fence"] = all_data["Fence"].fillna("NA")
all_data["GarageQual"] = all_data["GarageQual"].fillna("NA")
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna("NA")
all_data["GarageCond"] = all_data["GarageCond"].fillna("NA")
all_data["GarageFinish"] = all_data["GarageFinish"].fillna("NA")
all_data["GarageType"] = all_data["GarageType"].fillna("NA")
all_data["BsmtCond"] = all_data["BsmtCond"].fillna("NA")
all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("NA")
all_data["BsmtQual"] = all_data["BsmtQual"].fillna("NA")
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("NA")
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("NA")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])
all_data["Utilities"] = all_data["Utilities"].fillna(all_data["Utilities"].mode()[0])
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(all_data["BsmtHalfBath"].mode()[0])
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(all_data["BsmtFullBath"].mode()[0])
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(all_data["BsmtFinSF2"].mode()[0])

all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(all_data["BsmtFinSF1"].mode()[0])
all_data["GarageArea"] = all_data["GarageArea"].fillna(all_data["GarageArea"].mode()[0])
all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].mode()[0])
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(all_data["BsmtUnfSF"].mode()[0])
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(all_data["TotalBsmtSF"].mode()[0])
all_data["GarageCars"] = all_data["GarageCars"].fillna(all_data["GarageCars"].mode()[0])
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].mode()[0])
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])

#Replace all object data by numericals using one hot encoding
#data=pd.concat([train_df, test_df],axis=0)
objList = all_data.select_dtypes(include = "object").columns
print (objList)

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    i = 0
    for each in cols:
        #print (each)
        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)
        if i == 0: 
            print (dummies)
            i = i + 1
        df = pd.concat([df, dummies], axis=1)
    return df

#Before
all_data.shape
##(2919, 80)

#One hot encoding done
all_data = one_hot(all_data, objList) 
#After
all_data.shape
##(2919, 406)

#Dropping duplicates columns if any
all_data = all_data.loc[:,~all_data.columns.duplicated()]
all_data.shape

#Dropping the original columns that has data type object 
all_data.drop(objList, axis=1, inplace=True)
all_data.shape

#Separating the train and test data from all_data and appending SalePrice column to the train data
train_df = all_data.iloc[:1460,:]
test_df = all_data.iloc[1460 :,:]
train_df["SalePrice"] = x_saleprice

X_train = train_df.drop(["SalePrice"], axis = 1)
Y_train = train_df["SalePrice"]
X_test = test_df

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error

n_folds = 5
cv = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(X_train.values)

def test_model(model):   
    msle = make_scorer(mean_squared_log_error)
    rmsle = np.sqrt(cross_val_score(model, X_train, Y_train, cv=cv, scoring = msle))
    score_rmsle = [rmsle.mean()]
    return score_rmsle

def test_model_r2(model):
    r2 = make_scorer(r2_score)
    r2_error = cross_val_score(model, X_train, Y_train, cv=cv, scoring = r2)
    score_r2 = [r2_error.mean()]
    return score_r2

#1. Simple linear regression model
# we wil not use linear regression because its mean log score is coming negative
from sklearn.linear_model import LinearRegression
clf_linearReg = LinearRegression()
#rmsle_linearreg = test_model(clf_linearReg)
#print (rmsle_linearreg )

#2. Lasso regression
# we wil not use because its rmsle is coming negative
from sklearn.linear_model import Lasso
clf_lasso = Lasso(alpha=0.0001)
rmsle_lasso = test_model_r2(clf_lasso)
print (rmsle_lasso)

#3. Ridge Model
from sklearn.linear_model import Ridge
clf_ridge = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,
      random_state=None, solver='auto', tol=0.001)
rmsle_ridge = test_model(clf_ridge)
print (rmsle_ridge, test_model_r2(clf_ridge))
##[0.1562332068843027] [0.8350220277268043]

#4. Elastic Net
from sklearn.linear_model import ElasticNet
clf_elastic_net = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.9,
           max_iter=100, normalize=False, positive=False, precompute=False,
           random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
#clf_elastic_net.get_params()
rmsle_elastic_net = test_model(clf_elastic_net)
print (rmsle_elastic_net, test_model_r2(clf_elastic_net))
##[0.15524191588871572] [0.8355319656725282]

#5. SVM algorithm
from sklearn import svm
clf_svm = svm.SVC()
rmsle_svm = test_model(clf_svm)
print (rmsle_svm, test_model_r2(clf_svm))
##[0.42655965911720095] [-0.21383191346231967]

#6. Decision Tree 
from sklearn.tree import DecisionTreeRegressor
clf_dtR = DecisionTreeRegressor(max_depth=5, random_state=51)
rmsle_dtR = test_model(clf_dtR)
print (rmsle_dtR, test_model_r2(clf_dtR))
##[0.20087638073362202] [0.7569636210379455]

#7. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

clf_rFR = RandomForestRegressor(max_depth=5, random_state=51)
rmsle_rFR = test_model(clf_rFR)
print (rmsle_rFR, test_model_r2(clf_rFR))
##[0.17295960323425913] [0.8220478028104417]

#8. Adaboost regressor
from sklearn.ensemble import AdaBoostRegressor
clf_aBR = AdaBoostRegressor(random_state=51, n_estimators=1000)
rmsle_aBR = test_model(clf_aBR)
print (rmsle_aBR, test_model_r2(clf_aBR))
##[0.20880462846912984] [0.7886447451714611]

#9. BaggingClassifier
from sklearn.ensemble import BaggingRegressor
clf_bgr = BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False,
                 max_features=1.0, max_samples=1.0, n_estimators=100,
                 n_jobs=None, oob_score=False, random_state=51, verbose=0,
                 warm_start=False)
rmsle_bgr = test_model(clf_bgr)
print (rmsle_bgr, test_model_r2(clf_bgr))
##[0.14719806924931875] [0.8539938038399392]

#10. XGboost
import xgboost as xgb
from xgboost import plot_importance

clf_xgb = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.5, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=10,
             min_child_weight=7, missing=None, monotone_constraints=None,
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
rmsle_xgb = test_model(clf_xgb)
print (rmsle_xgb, test_model_r2(clf_xgb))
##[0.13372606548796895] [0.8742884283414025]

#11. GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
clf_ggr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.15, loss='ls', max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=60,
                          min_weight_fraction_leaf=0.0, n_estimators=1500,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
                        
rmsle_ggr = test_model(clf_ggr)
print (rmsle_ggr, test_model_r2(clf_ggr))
##[0.14187482285610134] [0.8865067659268699]

#Training model (here I have just shown training with GradientBoostingRegressor but each time we can consider new model and train it
clf_ggr.fit(X_train, Y_train)
Y_pred = clf_ggr.predict(test_df) 

#Submission
pred=pd.DataFrame(Y_pred)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
print("Your submission was successfully saved!")

#Hyper parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#For GradientBoostingRegressor
param_grid = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 
              'min_samples_leaf':[1,3,5,7,9, 15, 20, 25, 30, 40, 50],
              'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1],
              'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001, 0.2], 
              'n_estimators':[10, 30, 50, 100,250,500,750,1000,1250,1500,1750],
              'max_features' : ['sqrt']
             }

asdf = GradientBoostingRegressor()

#clf = GridSearchCV(asdf, param_grid=param_grid, scoring='r2', n_jobs=-1)
clf = RandomizedSearchCV(asdf, param_grid, scoring='r2', n_jobs=-1)
 
clf.fit(X_train, Y_train)

print(clf.best_estimator_)

#Feature Importance chart
from matplotlib import pyplot
pyplot.bar(range(len(clf_ggr.feature_importances_)), clf_ggr.feature_importances_)
pyplot.show()

##Remove features with importance <0.00
for feat, importance in zip(train_df.columns, clf_ggr.feature_importances_):
    #print ('feature: ', feat, ' importance: ',importance)
    if importance <= 0.000 and feat != "SalePrice":
        train_df.drop([feat], axis = 1, inplace = True)
        test_df.drop([feat], axis = 1, inplace = True)

print (test_df.shape, train_df.shape)

#Now again retrain and generate Y_pred for submission