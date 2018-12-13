import pandas as pd
import numpy as np
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError
import numpy as np
from scipy import stats
import random
#from astroML.density_estimation import bayesian_blocks

def get_categorical_column_indexes(df, subset_cols = None, ignore_cols = None, threshold = 10):
    '''
    Function that returns categorical columns indexes from the dataFrame
    Input:
        df: pandas dataFrame
        subset_cols: list of columns to filter categorical columns from
        ignore: list of columns to ignore from categorical columns
    returns:
        list with indexes of the Categorical columns from the list of dataframe columns
    '''
    def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()+[x for x in df.select_dtypes(include=[np.number]).columns if df[x].nunique() < threshold]
    if subset_cols is not None and type(subset_cols) == list:
        cat_cols = [x for x in cat_cols if x in subset_cols]
    if ignore_cols is not None and type(ignore_cols) == list:
        cat_cols = [x for x in cat_cols if x not in ignore_cols]
    return column_index(df, cat_cols)

def get_numerical_column_indexes(df, subset_cols = None, ignore_cols = None, ignore_threshold = 10):
    '''
    Function that returns numerical columns indexes from the dataFrame
    Input:
        df: pandas dataFrame
        subset_cols: list of columns to filter categorical columns from
        ignore: list of columns to ignore from categorical columns
    returns:
        list with indexes of the Categorical columns from the list of dataframe columns
    '''
    def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    cat_cols = [x for x in df.select_dtypes(include=[np.number]).columns if df[x].nunique() >= threshold]
    if subset_cols is not None and type(subset_cols) == list:
        cat_cols = [x for x in cat_cols if x in subset_cols]
    if ignore_cols is not None and type(ignore_cols) == list:
        cat_cols = [x for x in cat_cols if x not in ignore_cols]
    return column_index(df, cat_cols)


class Encoding(BaseEstimator):
    categorical_columns = None
    return_df = False
    random_state = 30
    threshold = 50

    def __init__(self):
        pass

    def convert_input(self, X):
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, list):
                X = pd.DataFrame(np.array(X))
            elif isinstance(X, (np.generic, np.ndarray, pd.Series)):
                X = pd.DataFrame(X)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))
            X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        x = X.copy(deep = True)
        return x

    def get_categorical_columns(self, X):
        return X.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_numerical_columns(self,X):
        temp_x=X[X.columns[X.nunique()<=self.threshold]]
        col_names=temp_x.columns[temp_x.dtypes!='object']
        return col_names

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            if col in encoding_dict:
                freq_dict = encoding_dict[col]
                X[col] = X[col].apply(lambda x: freq_dict[x] if x  in freq_dict else np.nan)
        return X

    def create_encoding_dict(self, X, y):
        return {}

    def fit(self, X, y=None):
        if X is None:
            raise ValueError("Input array is required to call fit method!")
        X = self.convert_input(X)
        self.encoding_dict = self.create_encoding_dict(X, y)
        return self

    def transform(self, X):
        df = self.apply_encoding(X, self.encoding_dict)
        if self.return_df:
            return df
        else:
            return df.values

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = self.convert_input(X)
        for col in self.categorical_columns:
            freq_dict = self.encoding_dict[col]
            for key, val in freq_dict.iteritems():
                X.loc[X[col] == val, col] = key
        if self.return_df:
            return X
        else:
            return X.values

class DummyEncoding(Encoding):
    '''
    class to perform DummyEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''
    def __init__(self, categorical_columns = None, return_df = 'False'):
        self.categorical_columns = categorical_columns
        self.return_df = return_df

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            encoding_dict.update({col: X[col].astype('object').unique().tolist()})
        return encoding_dict

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            for val in encoding_dict[col]:
                col_val = '%s_%s'%(str(col), str(val))
                X[col_val] = X[col].apply(lambda x: 1 if x == val else 0)
            del X[col]
        return X

class ThresholdEncoding(Encoding):
    def __init__(self, categorical_columns = None, return_df = False, threshold = 1, others='others'):
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.threshold = threshold
        self.others = others

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            if col in encoding_dict:
                freq_dict = encoding_dict[col]
                X[col] = X[col].apply(lambda x: freq_dict[x] if x in freq_dict else self.others)
        return X

    def create_encoding_dict(self, X, y=None):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            if type(self.threshold) == float:
                vc = X[col].value_counts(normalize = True).to_dict()
            else:
                vc = X[col].value_counts(normalize = False).to_dict()
            encoding_dict.update({col: {k:k for k,v in vc.iteritems() if v > self.threshold}})
        return encoding_dict

class Binning(Encoding):

    def __init__(self,categorical_columns=None,threshold=50,fp_rate_binning=0.05,return_df=False):
        self.categorical_columns=categorical_columns
        self.threshold=threshold
        self.fp_rate_binning=fp_rate_binning
        self.return_df=return_df

    def create_encoding_dict(self,X,y):

        encoding_dict={}
        try:
            if self.categorical_columns is None:
                self.categorical_columns = self.get_numerical_columns(X)
                temp_x=X[X.columns[X.nunique()<=self.threshold]]
            else:
                temp_x=X[self.categorical_columns]
            for i in self.categorical_columns:
                # print "Binning column:",i
                bins=bayesian_blocks(fitness='events',t=temp_x[i],p0 = self.fp_rate_binning)
                encoding_dict[i]=bins
        except:
            pass
        return encoding_dict

    def apply_encoding(self,X_in,encoding_dict):
        X = self.convert_input(X_in)
        temp_x=X[self.categorical_columns]
        # temp_x=X[X.columns[X.nunique()<=self.threshold]]
        for i in encoding_dict.keys():
            if i in temp_x.columns:
                X['binned_%s'%str(i)]=np.digitize(x=temp_x[i],bins=encoding_dict[i])
        return X

class FreqeuncyEncoding(Encoding):
    '''
    class to perform FreqeuncyEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''
    def __init__(self, categorical_columns = None, return_df = False):
        self.categorical_columns = categorical_columns
        self.return_df = return_df

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            encoding_dict.update({col: X[col].value_counts(normalize = True).to_dict()})
        return encoding_dict


class BaseNEncoding(Encoding):
    '''
    class to perform BaseNEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    base: base number
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''
    def __init__(self, categorical_columns = None, base = 2, return_df = False, delete_original_columns=True):
        if base<1 or base>10:
            raise ValueError("Either base is less than 1 or greater than 10 or n is less than 0")
        self.base = base
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.delete_original_columns = delete_original_columns

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            encoding_dict.update({col:{x:i for i,x in enumerate(pd.unique(X[col]))}})
        return encoding_dict

    def toStrOfBase(self, n, base):
        convertString = "0123456789"
        if n == 0: return '0'
        if base == 1: return ''.join(['1' if i == 0 else '0' for i in range(n)])
        if n < base:
            return str(n)
        else:
            return self.toStrOfBase(n//base,base) + convertString[n%base]

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            freq_dict = encoding_dict[col]
            _max = max(freq_dict.values())
            _max_base_len = len(self.toStrOfBase(_max, self.base))
            X['tmp_'+str(col)] = X[col].apply(lambda x: self.toStrOfBase(freq_dict[x], self.base).zfill(_max_base_len))
            for i in range(_max_base_len):
                X['col_%s_base%s_%d'%(str(col), str(self.base), i)] = X['tmp_'+str(col)].str[_max_base_len-i-1]
            del X['tmp_'+str(col)]
            if self.delete_original_columns:
                del X[col]
        return X


class TargetEncoding(Encoding):
    '''
    class to perform TargetEncoding on Categorical Variables
    Initialization Variabes:
    folds: no of folds when used fit_transform on the same dataset, default is 2
    random_state : random seed for KFolds

    '''
    def __init__(self, categorical_columns = None, folds = 2, random_state=30, return_df = False, regularize = True):
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.folds = folds
        self.random_state = random_state
        self.regularize = regularize

    def regularize_func(self, n_tot_good, n_tot_bad, n_att_good, n_att_bad):
        def lamda_n_fun(n, k, f=None):
            '''
            returns
            lambda(n)=1/(1+e**(-(n-k)/f))
            '''
            return k/float(n)
            return 1.0/(1+np.exp(-1*((n-k)/float(f))))

        lam_n = lamda_n_fun(n_tot_good + n_tot_bad, n_att_good + n_att_bad)
        return ((lam_n*(n_att_good/float(n_att_good+n_att_bad))) + (1.0-lam_n)*(n_tot_good/float(n_tot_good+n_tot_bad)))

    def __calculate_mean_value(self, x, y, val_indexes):
        try:
            n_att_good, n_att_bad = np.count_nonzero(y[val_indexes] == 0), np.count_nonzero(y[val_indexes] == 1)
            n_tot_good, n_tot_bad = np.count_nonzero(y == 0), np.count_nonzero(y == 1)
            if self.regularize == True:
                return self.regularize_func(n_tot_good, n_tot_bad, n_att_good, n_att_bad)
            return n_att_good/float(n_att_good+n_att_bad)
        except Exception as e:
            return np.nan

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            enc_d = {}
            for c_val in pd.unique(X[col]):
                val_indexes = np.where(X[col] == c_val)
                enc_d.update({c_val: self.__calculate_mean_value(X[col], y.iloc[:,0].values, val_indexes)})
            encoding_dict.update({col:enc_d})
        return encoding_dict

    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError("Input and Output variable is required to call fit method!")
        X = self.convert_input(X)
        y = self.convert_input(y)
        self.encoding_dict = self.create_encoding_dict(X, y)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = self.convert_input(X)
        y = self.convert_input(y)
        kf = KFold(n_splits= self.folds, random_state=self.random_state)
        copy_X = deepcopy(X)
        for i, (train_index, test_index) in enumerate(kf.split(copy_X)):
            encoding_dict = self.create_encoding_dict(X.iloc[train_index], y.iloc[train_index])
            copy_X.iloc[test_index] = self.apply_encoding(X.iloc[test_index], encoding_dict)
        if self.return_df:
            return copy_X
        else:
            return copy_X.values


class WoeEncoding(Encoding):
    '''
    WEIGHT OF EVIDENCE | INFORMATION VALUES ENCODING
    class to perform Weight Of Evidence Encoding on Categorical Variables
    Initialization Variabes:
    cat_cols_index: list of indexes for categorical columns from the dataframe
    folds: no of folds when used fit_transform on the same dataset, default is 2
    random_state : random seed for KFolds
    '''
    def __init__(self, categorical_columns = None, folds = 2, random_state=30, return_df = False):
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.folds = folds
        self.random_state = random_state


    def __calculate_woe(self, x, y, val_indexes, b = 1.0):
        '''
        Weight of Evidence is computed as (Distribution of Good Credit Outcomes)/(Distribution of Bad Credit Outcomes)
        Or the ratios of Distr Goods / Distr Bads for short,
        where Distr refers to the proportion of Goods or Bads in the respective group,
        relative to the column totals, i.e., expressed as relative proportions of the total number of Goods and Bads.
        '''
        try:
            n_att_good, n_att_bad = np.count_nonzero(y[val_indexes] == 0), np.count_nonzero(y[val_indexes] == 1)
            n_tot_good, n_tot_bad = np.count_nonzero(y == 0), np.count_nonzero(y == 1)
            distr_good, distr_bad = ((n_att_good * b) / n_tot_good), ((n_att_bad * b) / n_tot_bad)
            if distr_good == 0:
                woe = -20
            elif distr_bad == 0:
                woe = 20
            else:
                woe = np.log(distr_good/distr_bad)
            iv = (distr_good - distr_bad) *  woe
            return woe, iv
        except Exception as e:
            return np.nan, 0

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        information_values = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            enc_d, iv = {}, 0
            for c_val in pd.unique(X[col]):
                val_indexes = np.where(X[col] == c_val)
                woe, info_val = self.__calculate_woe(X[col], y.iloc[:,0].values, val_indexes)
                iv+= info_val
                enc_d.update({c_val: woe})
            encoding_dict.update({col:enc_d})
            information_values.update({col:iv})
        return encoding_dict, information_values

    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError("Input and Output variable is required to call fit method!")
        X = self.convert_input(X)
        y = self.convert_input(y)
        self.encoding_dict, self.information_values = self.create_encoding_dict(X, y)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        kf = KFold(n_splits= self.folds, random_state=self.random_state)
        X = self.convert_input(X)
        y = self.convert_input(y)
        copy_X = deepcopy(X)
        for i, (train_index, test_index) in enumerate(kf.split(copy_X)):
            encoding_dict, iv = self.create_encoding_dict(X.iloc[train_index], y.iloc[train_index])
            copy_X.iloc[test_index] = self.apply_encoding(X.iloc[test_index], encoding_dict)
        if self.return_df:
            return copy_X
        else:
            return copy_X.values


class BackwardDifferenceEncoding(Encoding):
    '''
    class to perform BackwardDifferenceEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''

    def __init__(self, categorical_columns = None):
        self.categorical_columns = categorical_columns


    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        if self.categorical_columns is None:
            self.categorical_columns = self.get_categorical_columns(X)
        for col in self.categorical_columns:
            encoding_dict.update({col: X[col].value_counts(normalize = True).to_dict()})
        return encoding_dict

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            freq_dict = encoding_dict[col]
            X['tmp_'+str(col)] = X[col].apply(lambda x: freq_dict[x])
            t = sorted(freq_dict.values())
            for i, (a, b) in enumerate(zip(t[:-1], t[1:])):
                d = b-a
                X['col_bd_%s_%d'%(str(col), i)] = X['tmp_'+str(col)].apply(lambda x: -1*(d) if x <= d else d)
            del X['tmp_'+str(col)]
            del X[col]
        return X

class CatbEncoding(object):
    '''
    num_iter: no of permutations of dataset(train) to be tried to get the encodings for cat_cols
    cat_cols: list of categorical columns; pass None to use all categorical columns
    prior: by default it is the ratio of 1's in the target variable similar to the implementation in catboost
    pass None for using default value of prior
    target: pd.Series
    train: pd.DataFrame required to obtain the encodings

    Eg.
    ce=CatbEncoding(100,None,None)
    ce.fit(train,target)
    ce.transform(train)
    '''

    def __init__(self,num_iter,cat_cols,prior):
        self.num_iter=num_iter
        self.cat_cols=cat_cols
        self.prior=prior
        self.encoding_dict=None

    def catb_encoding(self,train,target,num_iter,cat_cols,prior):
        train=train.reset_index(drop=True)
        num_iter=num_iter
        if prior is None:
            prior=target.mean()
        if num_iter is None:
            num_iter=100
        if cat_cols is None:
            cat_cols=train.select_dtypes(['object']).columns.values.tolist()

        df=copy.deepcopy(train)
        df['target_val']=target.values
        indices=df.index.values

        encoding_dict={}

        for col in cat_cols:
            cat_val_final=[]

            for i in range(num_iter):
                random.seed(i)
                random.shuffle(indices)
                df=df.iloc[indices,:]
                li=[]
                li=(df.groupby(col)['target_val'].cumsum()-df['target_val'] + prior)/(df.groupby(col).cumcount()+1).values.tolist()
                cat_val_final.append(pd.DataFrame({'cat_val':li,'indices':indices}).sort_values('indices').cat_val.values.tolist())

            cat_val_final=np.mean(cat_val_final,axis=0)
            map_dict=dict(pd.DataFrame({'col':df[col],'cat_val':cat_val_final}).groupby('col')['cat_val'].mean())
            encoding_dict.update({col:map_dict})
        return encoding_dict

    def fit(self,train,target):
        self.encoding_dict=self.catb_encoding(train,target,self.num_iter,self.cat_cols,self.prior)
        return self.encoding_dict

    def transform(self,x):
        if self.encoding_dict is None:
            raise NotFittedError('Call fit first')
        for i in self.encoding_dict.keys():
            x[i]=x[i].map(self.encoding_dict[i])
        return x
