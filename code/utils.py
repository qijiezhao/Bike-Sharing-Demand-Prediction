import os,sys
import numpy as np
import pandas as pd
import csv

train_set='../data/train.csv'
test_set='../data/test.csv'
label_name=['casual','registered','count']

'''
transfor the time stamps to six parts(0-3,4-7,8-11,12-15,16-19,20-23)
'''
def transform_num(time_str):
    hour_info=time_str.strip().split(' ')[-1].split(':')[0]
    num_part=int(hour_info)/4
    return num_part

'''
read data,flag=0:train-set flag=1:test-set
'''
def load_data(path=train_set,flag=0):
    train_data=pd.read_csv(path)
    if flag==0:
        labels_list=[train_data[name] for name in label_name]
        labels_list=np.array(labels_list).T
    train_data['datetime']=[transform_num(item) for item in train_data['datetime']]
    items=[item for item in train_data if not item in label_name]
    train_data=np.array([train_data[item] for item in items]).T
    if flag==0:
        return labels_list,train_data
    test_data=train_data
    return test_data
