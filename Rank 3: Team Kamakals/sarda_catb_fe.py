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

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

import catboost
from catboost import CatBoostClassifier,Pool,cv


train=pd.read_csv('train_LZdllcl.csv')
test=pd.read_csv('test_2umaH9m.csv')
sam_sub=pd.read_csv('sample_submission_M0L0uXE.csv')
target=train.is_promoted
train.drop(['employee_id','is_promoted'],axis=1,inplace=True)
del test['employee_id']
original_df=pd.concat([train,test])

cat_cols=original_df.columns[original_df.nunique()<35].tolist()
num_cols=[i for i in original_df.columns if i not in cat_cols]
categorical_indices=[original_df.columns.get_loc(i) for i in cat_cols]

catb_tot=copy.deepcopy(original_df)
catb_tot.reset_index(drop=True,inplace=True)
catb_tot.previous_year_rating=catb_tot.previous_year_rating.fillna(6.0)
catb_tot.education=catb_tot.education.fillna('missing_cat')

cat_num_feats=pd.DataFrame(np.column_stack([catb_tot[m[0]].map(dict(catb_tot.groupby(m[0])[m[1]].mean()))
 for m in [(a,b) for a in cat_cols for b in num_cols]]),
             columns=['cat_num_feat'+str(i) for i in range(len(cat_cols)*len(num_cols))])
catb_tot=pd.concat([catb_tot,cat_num_feats],axis=1)

train_catb=catb_tot.iloc[:train.shape[0]]
test_catb=catb_tot.iloc[train.shape[0]:]

categorical_indices=[catb_tot.columns.get_loc(i) for i in cat_cols]


folds=StratifiedKFold(n_splits=5,random_state=100,shuffle=True)
fold_score=[]
test_preds=[]
test_df_preds=[]
indexes=[]
for tr_index,test_index in folds.split(train_catb,y=target):
    xtrain= train_catb.iloc[tr_index]
    xval =train_catb.iloc[test_index]
    ytrain,yval=target.iloc[tr_index],target.iloc[test_index]    
    clf_catb = CatBoostClassifier(depth=6,iterations=5000,random_seed=100, od_type='Iter', od_wait=100, 
                                    eval_metric='AUC')
    clf_catb.fit(xtrain, ytrain, cat_features=categorical_indices,eval_set=(xval,yval), 
                 use_best_model=True,verbose=True)
    test_preds.append(clf_catb.predict_proba(xval)[:,1])
    test_df_preds.append(clf_catb.predict_proba(test_catb)[:,1])
    score=roc_auc_score(yval,clf_catb.predict_proba(xval)[:,1])
    fold_score.append(score)             
    indexes.append(xval.index)
print("SCORE:"),(np.mean(fold_score)), np.std(fold_score)

cv_preds=[j for i in test_preds for j in i]
cv_index=[j for i in indexes for j in i]
cv_preds=pd.DataFrame({'cv_preds':cv_preds,'cv_index':cv_index}).sort_values('cv_index').cv_preds
catb_preds=pd.Series(np.mean(test_df_preds,axis=0))

roc_auc_score(target,cv_preds)

a=np.linspace(0,1,100).tolist()
b=[f1_score(target,(cv_preds>i).astype('int')) for i in a]
a[np.argmax(b)]
pd.DataFrame({
        '24':( catb_preds>0.3030 ).astype('int'),
        '25':( catb_preds>0.2929 ).astype('int'),
        '26':( catb_preds>0.2626   ).astype('int'),
        '27':( catb_preds>0.2727).astype('int'),
        '28':( catb_preds>0.2828).astype('int')    
            }).to_csv('all_preds_bagged_catb_extra_feats_26_30.csv',index=False)

