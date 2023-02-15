import pandas as pd
import warnings
import numpy as np
from params import *
from pre_process import *
from predict import *
from plot import *

warnings.filterwarnings("ignore")

df_train = pd.read_csv('./dataset/train.csv', sep=',', encoding='utf-8')
df_test = pd.read_csv('./dataset/test.csv', sep=',', encoding='utf-8')
print("Train:{} Test:{}".format(df_train.shape,df_test.shape))

df_model_train = pre_process(df_train,'train')
df_model_test = pre_process(df_test,'test')
df_model_train,df_model_test = merge_columns(df_model_train,df_model_test)
print('Processing data completed!')

if MONITOR_MODEL:
    models,results = monitor_model(df_model_train)
    print("Monitor train:")
    for m in results.keys():
        print(m)
        print(np.round(np.array(results[m]),4))
    if PLOT_MODEL:
        df_pre = read_pre_data(results)
        plot_bar(df_pre)

selected_model_name = 'GBoost'
selected_model,train_acc = train_model(df_model_train,selected_model_name)
print("Done!")

if PRINT_RESULT:
    pre_result = predict(df_model_test,selected_model)
    print_pre_result(TXT_PATH,pre_result)
    print("Predict via selected {} completed!".format(selected_model_name))



