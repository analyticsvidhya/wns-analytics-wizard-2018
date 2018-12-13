get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import pandas as pd 
import numpy as np
pd.options.display.max_columns=100
pd.options.display.max_rows=500
import sys

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
import copy
from feature_encoding import FreqeuncyEncoding,CatbEncoding,TargetEncoding

train=pd.read_csv('train_LZdllcl.csv')
test=pd.read_csv('test_2umaH9m.csv')

print train.shape, test.shape


y=train.is_promoted
train.drop(['is_promoted'],axis=1,inplace=True)


train.drop(['employee_id'],axis=1,inplace=True)
test.drop(['employee_id'],axis=1,inplace=True)
df=pd.concat((train,test),axis=0)


cat_cols=df.columns[df.dtypes=='object'].tolist()


num_cols=['age', 'previous_year_rating','length_of_service','avg_training_score']

print cat_cols,num_cols

cat_num_feats=pd.DataFrame(np.column_stack([df[m[0]].map(dict(df.groupby(m[0])[m[1]].mean()))
 for m in [(a,b) for a in cat_cols for b in num_cols]]),
             columns=['cat_num_feat'+str(i) for i in range(len(cat_cols)*len(num_cols))])

df.reset_index(drop=True,inplace=True)
df=pd.concat((df,cat_num_feats),axis=1)


fe=FreqeuncyEncoding(categorical_columns=cat_cols,return_df=True)
df=fe.fit_transform(df)


fe_train=df.iloc[:train.shape[0],:]
fe_test=df.iloc[train.shape[0]:,:]


lgb_mods=[LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
         learning_rate=0.1, max_depth=-1, metric='None',
         min_child_samples=20, min_child_weight=4, min_split_gain=0.0,
         n_estimators=10000, n_jobs=8, num_leaves=30, objective=None,
         random_state=None, reg_alpha=0.0, reg_lambda=0.2, silent=True,
         subsample=0.7, subsample_for_bin=200000, subsample_freq=1),
          LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
         learning_rate=0.1, max_depth=-1, metric='None',
         min_child_samples=20, min_child_weight=12, min_split_gain=0.0,
         n_estimators=10000, n_jobs=8, num_leaves=30, objective=None,
         random_state=None, reg_alpha=0.0, reg_lambda=0.5, silent=True,
         subsample=0.8, subsample_for_bin=200000, subsample_freq=1),
          LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
         learning_rate=0.1, max_depth=-1, metric='None',
         min_child_samples=20, min_child_weight=14, min_split_gain=0.0,
         n_estimators=10000, n_jobs=8, num_leaves=30, objective=None,
         random_state=None, reg_alpha=0.0, reg_lambda=0.1, silent=True,
         subsample=0.7, subsample_for_bin=200000, subsample_freq=1),
          LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
         learning_rate=0.1, max_depth=-1, metric='None',
         min_child_samples=20, min_child_weight=16, min_split_gain=0.0,
         n_estimators=10000, n_jobs=8, num_leaves=30, objective=None,
         random_state=None, reg_alpha=0.0, reg_lambda=0.3, silent=True,
         subsample=0.8, subsample_for_bin=200000, subsample_freq=1),
          LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.6,
         learning_rate=0.1, max_depth=-1, metric='None',
         min_child_samples=20, min_child_weight=18, min_split_gain=0.0,
         n_estimators=10000, n_jobs=8, num_leaves=40, objective=None,
         random_state=None, reg_alpha=0.0, reg_lambda=0.3, silent=True,
         subsample=0.7, subsample_for_bin=200000, subsample_freq=1)
         ]


def lgb_f1(ytrue,preds):
    preds=(preds>f1_threshold).astype('int')
    #print preds
    return 'f1_score',f1_score(y_pred=preds,y_true=ytrue), True

def xgb_f1(preds,ytrue):
    y_tr=ytrue.get_label()
    preds=(preds>f1_threshold).astype('int')
    return 'f1_score',-1*f1_score(y_pred=preds,y_true=y_tr)

def preds(mod,df_train,y,df_test=None,rank=False,seed=100,is_feat_imp=False):
    """ is_feat_imp is a flag to get normalized feature importance only """

    folds=StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
    fold_score=[]
    test_preds=[]
    feat_imp=[]
    cv_preds=[]
    indexes=[]
    i=1
    f1_pos=[]
    for tr_index,test_index in folds.split(df_train,y=y):
        xtrain= df_train.iloc[tr_index]
        xval =df_train.iloc[test_index]
        ytrain,yval=y.iloc[tr_index],y.iloc[test_index]
        watchlist = [ (xval,yval)]
        if (isinstance(mod,LGBMClassifier)):
            mod.fit(X=xtrain,y=ytrain,eval_set=watchlist,early_stopping_rounds=100,verbose=False,eval_metric=lgb_f1)
            preds=mod.predict_proba(xval)[:,1]
        elif(isinstance(mod,XGBClassifier)):
            mod.fit(X=xtrain,y=ytrain,eval_set=watchlist,early_stopping_rounds=100,verbose=False,eval_metric=xgb_f1)
            preds=mod.predict_proba(xval)[:,1]
        else:
            mod.fit(xtrain,ytrain)
            preds=mod.predict_proba(xval)[:,1]
            score=f1_score(yval,preds)

        if isinstance(mod,XGBClassifier):
            score=mod.best_score
        if isinstance(mod,LGBMClassifier):
            score=mod.best_score_['valid_0']['f1_score']
        if rank==True:
            cv_preds.append(pd.Series(preds).rank())
            if df_test is None:
                pass
            else:
                test_preds.append(pd.Series(mod.predict_proba(df_test)[:,1]).rank())
        else:
            cv_preds.append(preds)
            if df_test is None:
                pass
            else:
                test_preds.append(mod.predict_proba(df_test)[:,1])
        indexes.append(test_index)
        if (((isinstance(mod,LogisticRegression))==False)&((isinstance(mod,KNeighborsClassifier))==False)):
            feat_imp.append(mod.feature_importances_)
        fold_score.append(score)
        print preds.mean()
        f1_pos.append(f1_maximizer(preds,yval))
    cv_preds=[j for i in cv_preds for j in i]
    cv_index=[j for i in indexes for j in i]
    final_cv_preds=pd.DataFrame({'cv_preds':cv_preds,'cv_index':cv_index}).sort_values('cv_index').cv_preds.values
    
    print fold_score,np.mean(fold_score)
    print 'avg f1 pos',np.mean(f1_pos)
    return final_cv_preds,test_preds

def f1_maximizer(pred,y):
    sol=pd.DataFrame({'true':y,'prob':pred}).sort_values('prob',ascending=0).reset_index(drop=True)
    n=sum(y==1)
    scores=[f1_score(y_pred=(sol.prob>sol.prob[i]).astype('int'),y_true=sol.true) for i in range(n)]
    print sol.prob[np.argmax(scores)], np.argmax(scores)/float(sol.shape[0]),np.max(scores)
    return np.argmax(scores)/float(sol.shape[0]) 

def threshold_optimizer(y_preds,y_true):
    sc=[]
    thres=np.linspace(0,1,100)
    for i in thres:
        temp_preds=(y_preds>i).astype('int')
        sc.append(f1_score(y_true=y_true,y_pred=temp_preds))
        print f1_score(y_true=y_true,y_pred=temp_preds), i
    print 'optimum thresh',thres[np.argmax(sc)],max(sc)
    

print fe_train.shape
vals=[0.26,0.27,0.28,0.29,0.3]
lgb2_test_preds=[]
for i in range(len(vals)):
    f1_threshold=vals[i]
    n=preds(df_train=fe_train,y=y,seed=100,df_test=fe_test,mod=lgb_mods[i])
    lgb2_test_preds.append((pd.Series(np.column_stack(n[1]).mean(axis=1))>f1_threshold).astype('int'))
    print (pd.Series(np.column_stack(n[1]).mean(axis=1))>f1_threshold).value_counts(1)
    print pd.Series(n[0]>f1_threshold).value_counts(1)


sols=pd.DataFrame(np.column_stack(lgb2_test_preds))

sols.to_csv('lgb_32_feats.csv',index=False)

