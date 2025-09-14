!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# Show all columns
from rdkit import Chem
from mordred import Calculator, descriptors
pd.set_option('display.max_columns', None)






tg=pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv')
tc=pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv')
rg=pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv')
ffv=pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv')
density=pd.read_csv('/kaggle/input/modred-dataset/desc_de.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id']





for i in (tg,tc,rg,ffv,density):
     i.drop(columns=[col for col in i.columns if i[col].nunique() == 1],axis=1,inplace=True)







# Remove columns with object or category dtype
tg = tg.select_dtypes(exclude=['object', 'category'])
rg = rg.select_dtypes(exclude=['object', 'category'])
ffv = ffv.select_dtypes(exclude=['object', 'category'])
tc = tc.select_dtypes(exclude=['object', 'category'])
density  = density.select_dtypes(exclude=['object', 'category'])





mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]

# Initialize the Mordred Calculator
calc = Calculator(descriptors, ignore_3D=True) # ignore_3D=True for 2D descriptors

desc_test = calc.pandas(mols_test)





def model(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    train_cols = set(train_d.columns) - {target}
    test_cols = set(test_d.columns)
   # Intersect the feature columns
    common_cols = list(train_cols & test_cols)
    X=train_d[common_cols].copy()
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model()
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d[common_cols].copy())
       return submission
    




model(tg,desc_test,CatBoostRegressor,'Tg',submission=False)
model(ffv,desc_test,CatBoostRegressor,'FFV',submission=False)
model(tc,desc_test,CatBoostRegressor,'Tc',submission=False)
model(density,desc_test,CatBoostRegressor,'Density',submission=False)
model(rg,desc_test,CatBoostRegressor,'Rg',submission=False)



sub={'id':ID,'Tg':model(tg,desc_test,CatBoostRegressor,'Tg',submission=True),
     'FFV':model(ffv,desc_test,CatBoostRegressor,'FFV',submission=True),
     'Tc':model(tc,desc_test,CatBoostRegressor,'Tc',submission=True),
     'Density':model(density,desc_test,CatBoostRegressor,'Density',submission=True),
     'Rg':model(rg,desc_test,CatBoostRegressor,'Rg',submission=True)}


submission=pd.DataFrame(sub)
submission.to_csv('submission.csv',index=False)