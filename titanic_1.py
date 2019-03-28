def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.exceptions import DataConversionWarning

import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as TTS, RandomizedSearchCV as RSC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.metrics import f1_score as F1
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from prince import PCA, MCA
pd.options.mode.chained_assignment = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_id = test['PassengerId']
train_len = train.shape[0]
df = pd.concat([train, test], axis = 0, sort= True)

df['Name'] = [x.split(', ')[1].split('.')[0] for x in df.Name]
df['Name'] = df.Name.replace(['Mme', 'Mlle', 'Ms'], 'Miss')
df['Name'] = df.Name.replace('Lady', 'Mrs')
df['Name'] = df.Name.replace(['Jonkheer', 'the Countess', 'Major', 'Col', 'Capt', 'Rev', 'Don', 'Dona', 'Sir', 'Dr', 'Master'], 'Hierarchical')

df['Cabin'] = [str(x)[0] for x in df.Cabin]
df['Cabin'] = df.Cabin.replace('T', 'A')
df['Cabin'] = df.Cabin.replace('n', np.nan)
df['Cabin_temporary'] = df['Cabin']

df['Fare'].fillna(df['Fare'].mean(), inplace = True)
df['Embarked'].fillna('Q', inplace = True)

grouping = df['Ticket'].value_counts().to_dict()
df['Ticket'] = [grouping[x] for x in df['Ticket']]
df.loc[df['Ticket'] > 1, 'Ticket'] = 'Group'
df.loc[df['Ticket'] == 1, 'Ticket'] = 'Individual'

train = df[:train_len]
test = df[train_len:]

train['Age'] = train.groupby(['Embarked', 'Pclass', 'Ticket'])['Age'].apply(lambda x: x.fillna(x.mean()))
test['Age'] = test.groupby(['Embarked', 'Pclass', 'Ticket'])['Age'].apply(lambda x: x.fillna(x.mean()))

train['AgeBin'] = np.searchsorted([12, 24, 36, 48, 60], train['Age']).astype(np.int64)
test['AgeBin'] = np.searchsorted([12, 24, 36, 48, 60], test['Age']).astype(np.int64)

def classification(groups, data, target, copied_target):
    ## This function fills in the missing variables in category features via group-by.
    groups = groups + [target]
    length = list(range(len(groups) - 1))
    
    grouping = data.groupby(groups, as_index = False)[copied_target].count()
    grouping = grouping.groupby(groups[:-1]).apply(lambda x: x[target][x[copied_target].idxmax()])
    grouping = grouping.to_frame().reset_index(level = length).rename(columns = {0: 'Class'})
    
    X = pd.merge(data, grouping, how = 'left', on = groups[:-1])
    X[target].fillna(X['Class'], inplace = True)
    X.drop(columns = ['Class'], axis = 1, inplace = True)
    
    return X

train = classification(['Embarked', 'Pclass', 'Ticket', 'AgeBin'], train, 'Cabin', 'Cabin_temporary')
train = classification(['Embarked', 'Pclass'], train, 'Cabin', 'Cabin_temporary')
train = classification(['Embarked'], train, 'Cabin', 'Cabin_temporary')

test = classification(['Embarked', 'Pclass', 'Ticket', 'AgeBin'], test, 'Cabin', 'Cabin_temporary')
test = classification(['Embarked', 'Pclass'], test, 'Cabin', 'Cabin_temporary')
test = classification(['Embarked'], test, 'Cabin', 'Cabin_temporary')

# if TypeError: "float found", means there's likely still null values in Cabin features.
##train['Cabin'] = [ord(x) - 64 for x in train.Cabin]
##test['Cabin'] = [ord(x) - 64 for x in test.Cabin]
train.drop(columns = ['Cabin_temporary'], axis = 1, inplace = True)
test.drop(columns = ['Cabin_temporary'], axis = 1, inplace = True)

class KMeans_Feature():
    def __init__(self, n_clusters = 2):
        self.n_clusters = n_clusters

    def fit(self, X, y = None):
        self.km = KMeans(n_clusters = self.n_clusters, random_state = 42)
        self.km.fit(X)
        return self

    def transform(self, X):
        centroid = self.km.labels_
        C = pd.DataFrame({'Centroid' : centroid})
        length = list(range(self.n_clusters))

        dictionary = {}
        for _ in length:
            dictionary[_] = chr(_+65)
        C = C.replace({'Centroid': dictionary})    
        return C
    
numeric_columns =  train.drop(columns = 'Survived').select_dtypes(exclude = ['object']).columns
categoric_columns =  train.drop(columns = 'Survived').select_dtypes(include = ['object']).columns

df_km = KMeans_Feature().fit(train[numeric_columns]).transform(train[numeric_columns])
train = pd.concat([train, df_km], axis = 1, sort = True)
df_km = KMeans_Feature().fit(test[numeric_columns]).transform(test[numeric_columns])
test = pd.concat([test, df_km], axis = 1, sort = True)


y = train.Survived
x = train.drop(columns = ['Survived'])
xtrain, xval, ytrain, yval = TTS(x, y, test_size = 0.3, random_state = 42, stratify = y)

categoric_transformer = Pipeline(steps = [('MCA', MCA(n_components = 2))])

preprocessor = ColumnTransformer(transformers = [('cat', categoric_transformer, categoric_columns)])
               
pipe = Pipeline(steps = [('preprocessor', preprocessor),
                                        ('Scaler', StandardScaler()),
                                        ('PCA', KernelPCA(n_components = 4, kernel = 'rbf')),
                                        ('XGB', xgb.XGBClassifier())])

RSCparameter = {'XGB__n_estimators': st.randint (300, 2000), 'XGB__learning_rate': st.uniform (0.01, 0.1),
                    'XGB__gamma': st.uniform (0.01, 0.5), 'XGB__reg_alpha':st.uniform (0.01, 0.5),
                    'XGB__max_depth': st.randint (3, 10), 'XGB__min_child_weight': st.randint (3, 10),
                    'XGB__subsample': [0.5, 0.6, 0.7], 'XGB__colsample_bytree': [0.5, 0.6, 0.7]}
          
model = RSC(pipe, RSCparameter,  scoring = 'f1', n_iter = 60, cv = 5, return_train_score = True)
model.fit(xtrain, ytrain)

y_val_pred = model.predict(xval).astype(int)
print('CV Train F1: %s' % (max(model.cv_results_['mean_train_score'])))
print('CV Validation F1: %s' % (max(model.cv_results_['mean_test_score'])))
print('Validation F1: %s' %(F1(yval, y_val_pred)))

parameter = []
for x, y in model.best_params_.items():
	new = ''.join(" '{}' : {}".format(x.rpartition('__')[-1], y))
	parameter.append(new)
parameter = ', '.join(parameter)
print(parameter)                
                
xtest = test.drop(columns = ['Survived'])
prediction = model.predict(xtest).astype(int)
sub = pd.DataFrame({'PassengerId' : test_id, 'Survived' : prediction})
sub.to_csv('submission_1.csv', index= False)

a = model.best_estimator_.named_steps['XGB'].feature_importances_
print(a)
