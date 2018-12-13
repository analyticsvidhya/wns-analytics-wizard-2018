from  __future__ import division

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
import numpy as np
import pandas as pd
import operator
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import re
import xgboost
from xgboost import XGBClassifier
import copy
import time
import warnings
warnings.simplefilter('ignore')
import sklearn
pd.options.display.max_columns = 1000
pd.options.display.max_seq_items = 1000

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
import copy

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

train=pd.read_csv('train_LZdllcl.csv')
test=pd.read_csv('test_2umaH9m.csv')
sam_sub=pd.read_csv('sample_submission_M0L0uXE.csv')
target=train.is_promoted
train.drop(['employee_id','is_promoted'],axis=1,inplace=True)
del test['employee_id']
original_df=pd.concat([train,test])
original_df.reset_index(drop=True,inplace=True)


cat_cols=original_df.columns[original_df.nunique()<35].tolist()
num_cols=[i for i in original_df.columns if i not in cat_cols]


cat_num_feats=pd.DataFrame(np.column_stack([original_df[m[0]].map(dict(original_df.groupby(m[0])[m[1]].mean()))
 for m in [(a,b) for a in cat_cols for b in num_cols]]),
             columns=['cat_num_feat'+str(i) for i in range(len(cat_cols)*len(num_cols))])


original_df=pd.concat([original_df,cat_num_feats],axis=1)
del cat_num_feats

original_df.shape

for i in cat_cols:
    original_df[i]=LabelEncoder().fit_transform(original_df[i])
for i in cat_cols:
    original_df[i]=original_df[i].map(dict(original_df[i].value_counts(1)))
train_lgb=original_df.iloc[:train.shape[0]]
test_lgb=original_df.iloc[train.shape[0]:]

def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.26).astype('int')).values)
    return 'f_1',score

def get_predictions(train,target,test,mod,mod_type,rand_state,is_shuffle,sample_weight,cat_feats):
        '''
        test: 0 if preds on test set aren't required
        test:pd.DataFrame if preds on test set are required
        '''
        folds=StratifiedKFold(n_splits=5,random_state=rand_state,shuffle=is_shuffle)
        fold_score=[]
        indexes=[]

        temp_train=copy.deepcopy(train)
        y=target

        cv_preds=[]
        preds=[]

        for i,(train_index,test_index) in enumerate(folds.split(temp_train,y)):

            xtrain= temp_train.iloc[train_index]
            xval =temp_train.iloc[test_index]
            ytrain,yval=y.iloc[train_index],y.iloc[test_index]
            indexes.append(xval.index)
            if (mod_type=='lr')|(mod_type=='nb')|(mod_type=='knn'):
                scaler=RobustScaler()
                xtrain=pd.DataFrame(scaler.fit_transform(xtrain),columns=temp_train.columns)
                xval=pd.DataFrame(scaler.transform(xval),columns=temp_train.columns)
            watchlist = [(xtrain, ytrain),(xval,yval)]
            try:
                train_sample_wt=sample_weight.iloc[train_index].values
                test_sample_wt=sample_weight.iloc[test_index].values
            except:
                train_sample_wt=None
            if (mod_type=='rf') |(mod_type=='lr')|(mod_type=='et')|(mod_type=='nb')|(mod_type=='knn'):
                if mod_type=='rf':
                    mod.fit(xtrain,ytrain,sample_weight=train_sample_wt)
                else:
                    mod.fit(xtrain,ytrain)
                preds_val=mod.predict_proba(xval)[:,1]
                score=roc_auc_score(yval,preds_val)
            else:                
                if mod_type=='xgb':
                    mod.fit(X=xtrain,y=ytrain,eval_set=watchlist,eval_metric=custom_f1,
                        early_stopping_rounds=100,verbose=False,sample_weight=train_sample_wt) 
                    score=-mod.best_score
                elif mod_type=='lgb':
                    mod.fit(X=xtrain,y=ytrain,eval_set=watchlist,eval_metric='auc',
                        early_stopping_rounds=100,verbose=False,sample_weight=train_sample_wt,
                           categorical_feature=cat_feats) 
                    score=mod.best_score_['valid_1']['f_1']
            print 'fold',i+1, 'score:',score
            fold_score.append(score)

            if isinstance(test,pd.DataFrame):
                if (mod_type=='lr')|(mod_type=='nb')|(mod_type=='knn'):
                    test=pd.DataFrame(scaler.transform(test),columns=temp_train.columns)
                preds_test=mod.predict_proba(test)[:,1]
                preds.append(preds_test)

            cv_preds.append(mod.predict_proba(xval)[:,1])

        print 'mean cv score:',(np.mean(fold_score))
        if isinstance(test,pd.DataFrame):
            test_df_preds=np.mean(preds,axis=0)
        else:
            test_df_preds=0

        cv_mod_preds=[j for i in cv_preds for j in i]
        cv_mod_index=[j for i in indexes for j in i]

        final_cv_preds=pd.DataFrame({'cv_preds':cv_mod_preds,'cv_index':cv_mod_index}).sort_values('cv_index').cv_preds.values
        return final_cv_preds,test_df_preds



#0.26 as threshold for custom f1 metric
def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.26).astype('int')).values)
    return 'f_1',score
best_model=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4,
       gamma=0.3, learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=4, missing=None, n_estimators=1500, nthread=15,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.9)
vv=get_predictions(train_lgb,target,test_lgb,best_model,'xgb',0,True,None,'auto')


#0.27 as threshold for custom f1 metric
def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.27).astype('int')).values)
    return 'f_1',score
best_model1=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=6, missing=None, n_estimators=1500, nthread=15,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.9)
vv1=get_predictions(train_lgb,target,test_lgb,best_model1,'xgb',0,True,None,'auto')

#0.28 as threshold for custom f1 metric
def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.28).astype('int')).values)
    return 'f_1',score
best_model2=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.6,
       gamma=0.5, learning_rate=0.1, max_delta_step=0, max_depth=6,
       min_child_weight=12, missing=None, n_estimators=1500, nthread=15,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.9)
vv2=get_predictions(train_lgb,target,test_lgb,best_model2,'xgb',0,True,None,'auto')


#0.25 as threshold for custom f1 metric
def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.25).astype('int')).values)
    return 'f_1',score
best_model3=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.6,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
       min_child_weight=4, missing=None, n_estimators=1500, nthread=15,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.9)
vv3=get_predictions(train_lgb,target,test_lgb,best_model3,'xgb',0,True,None,'auto')

#0.29 as threshold for custom f1 metric
def custom_f1(y_predicted,y_true):
    score= -f1_score(y_true.get_label(),((pd.Series(y_predicted)>0.29).astype('int')).values)
    return 'f_1',score
best_model4=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
       gamma=0.3, learning_rate=0.1, max_delta_step=0, max_depth=6,
       min_child_weight=14, missing=None, n_estimators=1500, nthread=15,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
vv4=get_predictions(train_lgb,target,test_lgb,best_model4,'xgb',0,True,None,'auto')


# ## This dataframe contains classes predicted at 5 different thresholds ranging from 0.25-0.29. The idea was to optimize xgboost classifier using hyperopt using a custom f1 score metric at thresholds ranging respectively from 0.25-0.29 which in turn will give 5 different models and 5 different set of predictions. 


pd.DataFrame({
                '26':  (vv[1]>0.26).astype('int'),
                '27':  (vv1[1]>0.27).astype('int'),
                '28':  (vv2[1]>0.28).astype('int'),
                '25':  (vv3[1]>0.25).astype('int'),
                '29':  (vv4[1]>0.29).astype('int'),
        
            }).to_csv('df_xgb_extra_feats_bagged_25_29.csv',index=False)
