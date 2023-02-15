import pandas as pd
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import numpy as np
from params import *

def read_pre_data(results):
    model_names = list(results.keys())
    stats = ['train_acc','test_acc','precision','recall','f1-score']
    data = []
    for i,m in enumerate(model_names):
        for j,st in enumerate(stats):
            data.append([m,st,results[m][j]])
    df_pre = pd.DataFrame(data,columns=['model','stat','result'])
    return df_pre

def plot_bar(df_pre):
    # plot
    fig = px.bar(
        df_pre,  
        x="model",  
        y="result",  
        color="stat",  
        text_auto=True,
        barmode='group', # barmode 设置为 group则为簇状柱形图，可选 stack(叠加)、group(并列)、overlay(覆盖)、relative(相对)
    )
    fig.update_layout(width = 800, height = 600)
    fig.update_layout(yaxis=dict(title=dict(standoff = 0.5,font={'size': 18})))
    fig.update_layout(xaxis=dict(title=dict(standoff = 3,font={'size': 18}),tickfont={'size': 16}))
    fig.update_layout(legend=dict(font={'size': 14},\
                                yanchor="top",y=0.99,\
                                xanchor="left",x=0.01,
                                bgcolor="rgba(0,0,0,0)"))
    fig.show()     