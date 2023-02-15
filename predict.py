import numpy as np
from params import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier

def monitor_model(df_model):
    X_train, y_train, X_test, y_test = gene_data(df_model,choice=["train_test",RANDOM_STATE,SPLIT_RATE])
    models = {}
    results = {}
    models['LR'],results['LR'] = LR(X_train, y_train, X_test, y_test)
    models['RF'],results['RF'] = RF(X_train, y_train, X_test, y_test)
    models['GBoost'],results['GBoost'] = GBoost(X_train, y_train, X_test, y_test)    
    models['XGBoost'],results['XGBoost'] = XGBoost(X_train, y_train, X_test, y_test) 
    print("StackV models:",list(models.keys()))
    models['Stacking'],results['Stacking'] = Stacking(X_train, y_train, X_test, y_test, models)       
    return models,results

def train_model(df_model,model_name,stack_models=None):
    X_train, y_train = gene_data(df_model,choice=["train"])
    if model_name == 'LR':
        model,train_acc = RF(X_train, y_train,[],[])
    elif model_name == 'RF':
        model,train_acc = LR(X_train, y_train,[],[])
    elif model_name == 'GBoost':
        model,train_acc = GBoost(X_train, y_train,[],[])
    elif model_name == 'XGBoost':
        model,train_acc = XGBoost(X_train, y_train,[],[])
    elif model_name == 'Stacking':
        model,train_acc = Stacking(X_train, y_train,[],[],stack_models)
    else:
        print('unknown model!')
        assert(0)
    return model,train_acc

def predict(df_model,model):
    X_test = gene_data(df_model,choice=["test"])
    return model.predict(X_test)

def print_pre_result(path,pre_result):
    f = open(path,mode='w')
    f.truncate(0)
    for y in pre_result:
        f.write(str(int(y))+'\n')
    f.close()

def gene_data(df_model,choice=["train_test",0,0.3]):
    if choice[0] == "train_test":
        rs = choice[1]
        ts = choice[2]
        x_df = df_model.drop(['target'],axis=1)
        y_df = df_model['target']
        X_train, X_test, y_train, y_test = train_test_split(x_df.values, y_df.values, test_size=ts, random_state=rs)
        return X_train, y_train, X_test ,y_test # note order change for subprocess
    elif choice[0] == "train":
        x_df = df_model.drop(['target'],axis=1)
        y_df = df_model['target']        
        X_train, y_train = x_df.values,y_df.values
        return X_train, y_train
    elif choice[0] == 'test':
        X_test = df_model.values
        return X_test
    else:
        print("unknown choice!")
        assert(0)

def get_score(y_test,y_pred,ave='macro'):
    precision = precision_score(y_test, y_pred, average=ave)
    recall = recall_score(y_test, y_pred, average=ave)
    f1 = f1_score(y_test, y_pred, average=ave)
    return precision,recall,f1

def LR(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression...")
    lrm = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=2, tol=0.5)
    lrm.fit(X_train, y_train)
    train_acc = lrm.score(X_train,y_train)
    if len(X_test):
        y_pred = lrm.predict(X_test)
        test_acc = lrm.score(X_test,y_test)
        precision,recall,f1 = get_score(y_test,y_pred)
        return lrm,[train_acc,test_acc,precision,recall,f1]
    else:
        return lrm,train_acc

def RF(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    rfc = RandomForestClassifier(n_estimators=100,random_state=50, max_features="auto")
    rfc.fit(X_train, y_train)
    train_acc = rfc.score(X_train,y_train)
    if len(X_test):
        y_pred = rfc.predict(X_test)
        test_acc = rfc.score(X_test,y_test)
        precision,recall,f1 = get_score(y_test,y_pred)
        return rfc,[train_acc,test_acc,precision,recall,f1]
    else:
        return rfc,train_acc

def GBoost(X_train, y_train, X_test, y_test):
    print("Training Gradient Boosting...")
    gbc = GradientBoostingClassifier(n_estimators=100)
    gbc.fit(X_train, y_train)
    train_acc = gbc.score(X_train,y_train)
    if len(X_test):
        y_pred = gbc.predict(X_test)
        test_acc = gbc.score(X_test,y_test)
        precision,recall,f1 = get_score(y_test,y_pred)
        return gbc,[train_acc,test_acc,precision,recall,f1]
    else:
        return gbc,train_acc

def XGBoost(X_train, y_train, X_test, y_test):
    print("Training XGBoosting...")
    xgb = XGBClassifier(n_estimators=100)
    xgb.fit(X_train, y_train)
    train_acc = xgb.score(X_train,y_train)
    if len(X_test):
        y_pred = xgb.predict(X_test)
        test_acc = xgb.score(X_test,y_test)
        precision,recall,f1 = get_score(y_test,y_pred)
        return xgb,[train_acc,test_acc,precision,recall,f1]
    else:
        return xgb,train_acc

def Stacking(X_train, y_train, X_test, y_test, models):
    print("Training Stacking...")
    lr = LogisticRegression()
    sclf = StackingCVClassifier(classifiers=list(models.values()), 
                            meta_classifier=lr,cv=len(models))
    
    sclf.fit(X_train, y_train)
    train_acc = sclf.score(X_train,y_train)
    if len(X_test):
        y_pred = sclf.predict(X_test)
        test_acc = sclf.score(X_test,y_test)
        precision,recall,f1 = get_score(y_test,y_pred)
        return sclf,[train_acc,test_acc,precision,recall,f1]
    else:
        return sclf,train_acc

