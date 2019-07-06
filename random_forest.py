# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 01:00:34 2018

@author: oo197
"""
import os
import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
np.random.seed(4590)

def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


num_rows = None
#load data sets
df_train = pd.read_csv('data/ivs2012.csv', nrows = num_rows )
#df_test = pd.read_csv('all/test.csv', nrows = num_rows )



df_train = reduce_mem_usage(df_train)
#df_test = reduce_mem_usage(df_test)


#deal with nan
nan_count = df_train.isnull().sum()
for df in [df_train]:
    #fill childage with 0
    for i in np.arange(1,7):  
        col_name = ('childage'+str(i))
        df[col_name].fillna(0,inplace=True)
    
    

#feature engineering, convert purchase_date into pd interpretable format
for df in [df_hist_trans,df_new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
    
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    

new_columns = get_new_columns('hist',aggs)
df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs) # .agg takes agg elements as inputs and and returns results
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group;gc.collect()

###process df_new_merchant_trans ###
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
    
new_columns = get_new_columns('new_hist',aggs)  #translate the agg items into column labels

df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days
df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']
df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')



#use hist_tran to find out most visited merchant for each card id
merchant_id_max = {}
df_hist_trans_group = df_hist_trans.groupby("card_id")
for card_id, group in df_hist_trans_group:
    #build a series that hosts the count of fav merchant
    sub_group = df_hist_trans_group.get_group(card_id)
    merchant_id_max[card_id] = sub_group.groupby("merchant_id").count().idxmax()["card_id"]
    
    ##aggregte merchant_id_max into train, test set
merchant_id_max = pd.DataFrame.from_dict(merchant_id_max, orient='index', columns=['merchant_id'])
merchant_id_max["card_id"] = merchant_id_max.index
df_train = df_train.merge(merchant_id_max, how='left', on='card_id')
df_test = df_test.merge(merchant_id_max, how='left', on='card_id')

# first attempt, remove nan columns from merchants to better fit customer data

df_merchants = df_merchants.drop_duplicates("merchant_id")

df_merchants = df_merchants.drop(columns = "avg_sales_lag3")
df_merchants = df_merchants.drop(columns = "avg_sales_lag6")
df_merchants = df_merchants.drop(columns = "avg_sales_lag12")
df_merchants = df_merchants.drop(columns = "category_2")


#encode catagorical features into numberical representasion
df_merchants['category_1'] = df_merchants['category_1'].map({'Y':1, 'N':0})
df_merchants['category_4'] = df_merchants['category_4'].map({'Y':1, 'N':0})


#leave merchant_group_id here for now, check existence later
#see how to check specific element value within columns

#map A to E for now, see if there's better way to deal with categories. Answer: get_dummies

df_most_recent_sales_range = pd.get_dummies(df_merchants['most_recent_sales_range'], prefix = "most_recent_sales_range")
df_merchants = pd.concat([df_merchants,df_most_recent_sales_range], axis = 1)
df_merchants = df_merchants.drop(columns = "most_recent_sales_range")
df_most_recent_purchases_range = pd.get_dummies(df_merchants['most_recent_purchases_range'], prefix = "most_recent_purchases_range")
df_merchants = pd.concat([df_merchants,df_most_recent_purchases_range], axis = 1)
df_merchants = df_merchants.drop(columns = "most_recent_purchases_range")

df_train = df_train.merge(df_merchants, how='left', on='merchant_id')
df_test = df_test.merge(df_merchants, how='left', on='merchant_id')


df_train = df_train.drop(columns = "merchant_id")
df_test = df_test.drop(columns = "merchant_id")

#delete the obsolete variables
del df_hist_trans_group;gc.collect()
del df_hist_trans;gc.collect()
del df_new_merchant_trans;gc.collect()
del df_merchants;gc.collect()
df_train.head(5)

### filter out the outliers ###
plt.figure(1)
plt.scatter(np.arange(df_train['target'].size),df_train['target'])
df_train['outliers'] = 0
df_train.loc[df_train['target'] < -30, 'outliers'] = 1
df_train['outliers'].value_counts()

for df in [df_train,df_test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']


### use outliers to map features, why????
for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['outliers'].mean()
    df_train[f] = df_train[f].map(order_label)
    df_test[f] = df_test[f].map(order_label)

nan_columns1 = df_train.isnull().sum().index[df_train.isnull().sum().astype(bool)].tolist()
df_train = df_train.drop(columns = nan_columns1)
df_test = df_test.drop(columns = nan_columns1)

nan_columns2 = df_test.isnull().sum().index[df_test.isnull().sum().astype(bool)].tolist()
df_train = df_train.drop(columns = nan_columns2)
df_test = df_test.drop(columns = nan_columns2)
    
### remove columns that shouldn't be used in training
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = df_train['target']
del df_train['target']  ## to reduce precessing time?






### K-fold splitting with stratification ###
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=4590)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = df_train.iloc[trn_idx][df_train_columns]
    val_data = df_train.iloc[val_idx][df_train_columns]

    clf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=4590,
                                min_samples_split  = 1000, min_samples_leaf =1000,
                                max_features = "sqrt", n_jobs = 6)
    clf.fit(trn_data, target.iloc[trn_idx])

    oof[val_idx] = clf.predict(val_data)
    
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[df_train_columns]) / folds.n_splits

predictions_train = clf.predict(df_train.iloc[:][df_train_columns])
score_rf1 = np.sqrt(mean_squared_error(predictions_train, target))

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))  
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False)) 
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)


# Extract single tree
estimator = clf.estimators_[0]


from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                filled=True, rounded=True,
                special_characters=True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o' 'tree.png'])


from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(clf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15 ,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 6, 8, 10, 12, 14]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

test_data = df_test[df_train_columns]
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 3, verbose=2, random_state=42, n_jobs = 6)
# Fit the random search model
no_sample_paramtest = 10000
train_labels = target[:no_sample_paramtest]
train_features = df_train.iloc[:no_sample_paramtest][df_train_columns]
rf_random.fit(train_features, train_labels)

pprint(rf_random.best_params_)


best_random = rf_random.best_estimator_
best_random.fit(train_features, train_labels)
predictions = best_random.predict(train_features)
score_random = np.sqrt(mean_squared_error(predictions, train_labels))



# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [50, 100, 200, 400],
    'max_features': ['sqrt'],
    'min_samples_leaf': [5, 10, 15, 20],
    'min_samples_split': [2,4,6],
    'n_estimators': [400,800,1600]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 6, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
pprint(grid_search.best_params_)

best_search = grid_search.best_estimator_
best_search.fit(train_features, train_labels)
predictions = best_search.predict(train_features)
score_search = np.sqrt(mean_squared_error(predictions, train_labels))


### K-fold splitting with stratification ###
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()
clf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=4590,
                                min_samples_split  = 1000, min_samples_leaf =1000,
                                max_features = "sqrt", n_jobs = 6)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = df_train.iloc[trn_idx][df_train_columns]
    val_data = df_train.iloc[val_idx][df_train_columns]

    clf.set_params(**grid_search.best_params_) 
    clf.fit(trn_data, target.iloc[trn_idx])

    oof[val_idx] = clf.predict(val_data)
    
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[df_train_columns]) / folds.n_splits

predictions_train = clf.predict(df_train.iloc[:][df_train_columns])
score_rf2 = np.sqrt(mean_squared_error(predictions_train, target))

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))  
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False)) 
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission1.csv", index=False)
