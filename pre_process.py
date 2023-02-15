import pandas as pd
import warnings
import re
import numpy as np
from sklearn import linear_model 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

warnings.filterwarnings("ignore")

def pre_process(df,table_name):
    df_mark = gene_mark_table(df,table_name)
    df_mark = trans_neighbourhood(df_mark)
    df_mark = trans_type(df_mark)
    df_mark = trans_insbook(df_mark)
    df_mark = trans_bathrooms(df_mark)
    df_mark = trans_bedrooms(df_mark)
    df_mark = complete_review(df_mark)
    df_mark = stan_scale(df_mark)
    df_model = gene_model_table(df_mark) #des&ame indoor
    return df_model

def gene_mark_table(df,table_name):
    '''
    mark_columns = ['description','amenities','longitude','latitude','neighbourhood','type','accommodates','bathrooms','bedrooms','instant_bookable',\
                'reviews','review_rating','review_scores_A','review_scores_B','review_scores_C','review_scores_D',\
                'target']
    if table_name == 'test':
        mark_columns = mark_columns[:-1] # delete target
    df_mark = df[mark_columns]
    '''
    df_mark = df.copy()
    return df_mark

def trans_neighbourhood(df_mark,neighbour_pattern = 'area'):
    #读取neighbour所固定的对应编号
    file = open("neighbourhood.txt",encoding = "utf-8").readlines()
    neigh_num = len(file)
    neighbour_hash = {} #{0:'Sydney',...}
    neighbour_num = {} #{'Sydney':0,...}
    for line in file:
        [index,nb] = line[:-1].split(',')
        neighbour_hash[int(index)] = nb
        neighbour_num[nb] = int(index)
    # neighbour trans func
    def neighbour_trans(x,trans_dict):
        return trans_dict[x]
    if not pd.api.types.is_int64_dtype(df_mark['neighbourhood']):
        if neighbour_pattern == 'area':
            areas = {0:[8,10,12,13,15,19,22,24,25,26,27,29,30,31,32,33,34,35,36],\
                    1:[0,1,2,6,9,11,16,17,18,20,21,23,28,37],\
                    2:[1,3,4,5,7,14]}
            neighbour_areas = {}
            for ar in areas.keys():
                for nb in areas[ar]:
                    neighbour_areas[neighbour_hash[nb]] = ar
            df_mark['neighbourhood'] = df_mark['neighbourhood'].apply(neighbour_trans, args=(neighbour_areas,))
        elif neighbour_pattern == 'num':
            df_mark['neighbourhood'] = df_mark['neighbourhood'].apply(neighbour_trans, args=(neighbour_num,))
        else:
            print('unknown pattern')
            assert 0
    return df_mark

def trans_type(df_mark):
    def type_trans(x,class_num):
        if class_num == 4:
            type_dict = {'Entire home/apt':0,'Private room':1,'Hotel room':2,'Shared room':3}
        elif class_num == 3:
            type_dict = {'Entire home/apt':0,'Private room':1,'Hotel room':2,'Shared room':2}
        return type_dict[x]  
    if not pd.api.types.is_int64_dtype(df_mark['type']):
        df_mark['type'] = df_mark['type'].apply(type_trans, args=(3,))
    return df_mark

def trans_insbook(df_mark):
    def ins_book_trans(x):
        return 1 if x == 't' else 0
    if not pd.api.types.is_int64_dtype(df_mark['instant_bookable']):
        df_mark['instant_bookable'] = df_mark['instant_bookable'].apply(ins_book_trans)
    return df_mark

def trans_bathrooms(df_mark):
    df_mark['bathrooms'][df_mark['bathrooms'].isnull()] = '1 bath'
    def bath_trans(x):
        num = x.split()[0]
        if len(num) > 4:
            return 0.0
        else:
            return float(num)    
    if not pd.api.types.is_float_dtype(df_mark['bathrooms']):
        df_mark['bathrooms'] = df_mark['bathrooms'].apply(bath_trans)
    return df_mark

def trans_bedrooms(df_mark):
    df_mark['bedrooms'][df_mark['bedrooms'].isnull()] = 1.0
    return df_mark

def complete_review(df_mark):

    # 补全reviews_score_A~D: 平均值
    mean_resc = []
    resc_valid_index = np.ones(len(df_mark)).astype(bool) #全True向量
    resc_labels = ['A','B','C','D']
    for i,rl in enumerate(resc_labels):
        valid_index = ~df_mark['review_scores_'+rl].isnull()
        resc_valid_index &= valid_index
        valid_scores = df_mark[valid_index]['review_scores_'+rl]
        mean_score = np.mean(valid_scores)
        df_mark['review_scores_'+rl][~valid_index] = mean_score
        mean_resc.append(mean_score)
    
    # 线性模型训练
    review_lm = linear_model.LinearRegression()
    df_scores = df_mark[resc_valid_index]
    X = df_scores[['review_scores_A','review_scores_B','review_scores_C','review_scores_D']].values
    y = df_scores['review_rating'].values
    review_lm.fit(X, y)
    print("Linera model:",review_lm.coef_,review_lm.intercept_)  #线性模型的系数和截距

    # 补全review_rating
    rera_valid_index = ~df_mark['review_rating'].isnull()
    if not rera_valid_index.all():
        tX = df_mark[['review_scores_A','review_scores_B','review_scores_C','review_scores_D']][~rera_valid_index].values
        ty = review_lm.predict(tX)
        df_mark['review_rating'][~rera_valid_index] = ty   

    return df_mark     

def stan_scale(df_mark):
    # 标准化已数值化的属性
    scaler = StandardScaler()
    num_cols = ['longitude', 'latitude', 'bathrooms', 'bedrooms', 'reviews', 'review_rating', 'review_scores_A', 'review_scores_B', 'review_scores_C', 'review_scores_D']
    scaler.fit(df_mark[num_cols].astype(float))
    df_mark[num_cols] = scaler.transform(df_mark[num_cols].astype(float))
    return df_mark

def trans_des_ame(df_mark):
    df_mark['description'][df_mark['description'].isnull()] = ''
    #n = 20
    # description
    all_des = df_mark['description'].values
    hanzi = re.compile(r'[\u4e00-\u9fa5]') 
    chinese_des = sum([1 for ad in all_des if hanzi.match(ad)])
    my_stopwords = list(ENGLISH_STOP_WORDS.union(['br']))
    #print(type(my_stopwords))
    des_tfm = TfidfVectorizer(stop_words=my_stopwords, min_df=0.1, use_idf=True, smooth_idf=True, norm=None)
    tf_des = des_tfm.fit_transform(all_des).toarray()
    tf_des_features = np.array(des_tfm.get_feature_names_out())
    #top_index = np.argsort(np.mean(tf_des,axis=0))[:n]
    #tf_des = tf_des[:,top_index]
    #tf_des_features = tf_des_features[top_index]
    # amenities
    all_ame = df_mark['amenities'].values
    ame_tfm = TfidfVectorizer(stop_words=my_stopwords, min_df=0.1, use_idf=True, smooth_idf=True, norm=None)
    tf_ame = ame_tfm.fit_transform(all_ame).toarray()
    tf_ame_features = np.array(ame_tfm.get_feature_names_out())
    #top_index = np.argsort(np.mean(tf_ame,axis=0))[:n]
    #tf_ame = tf_ame[:,top_index]
    #tf_ame_features = tf_ame_features[top_index]


    return tf_des,tf_ame,tf_des_features,tf_ame_features

def gene_model_table(df_mark):
    df_model = df_mark.copy()
    tf_des,tf_ame,tf_des_features,tf_ame_features = trans_des_ame(df_mark)
    df_model = df_model.drop(['reviews'],axis=1)
    #向量化处理 neighbourhood,type,accommodates
    df_model = pd.get_dummies(df_model, columns=['neighbourhood'])
    df_model = pd.get_dummies(df_model, columns=['type'])
    #文本数据tfidf导入
    df_model = df_model.drop(['description', 'amenities'], axis=1)
    des_columns = pd.DataFrame(tf_des, columns = tf_des_features)
    ame_columns = pd.DataFrame(tf_ame, columns = tf_ame_features)
    df_model = pd.concat([df_model, des_columns, ame_columns], axis=1)
    # 新增的 TF-IDF属性标准化
    scaler = StandardScaler()
    scaler.fit(df_model[tf_des_features].astype(float))
    df_model[tf_des_features] = scaler.transform(df_model[tf_des_features].astype(float))
    scaler.fit(df_model[tf_ame_features].astype(float))
    df_model[tf_ame_features] = scaler.transform(df_model[tf_ame_features].astype(float))
    return df_model

def merge_columns(df_train,df_test):
    '''
    df_train和df_test会因数据量不等发生列数不相同的情况，采取补全法使df_test的列数与df_train相同
    即从S(train列)-S(test列)中挑选差值个列赋值为全0补到df_test上
    完全不重复的列较难做到：
    1. des和ame中会有相同单词
    2. des和ame中会有bedrooms bathrooms
    '''
    diff_columns = list(set(df_train.columns) - set(df_test.columns) - set(['target']))
    sparse_columns = diff_columns[:len((df_train.columns))-len((df_test.columns))-1]
    print("Words not in test.csv:",diff_columns)
    for scol in sparse_columns:
        df_test[scol] = 0
    
    return df_train,df_test
