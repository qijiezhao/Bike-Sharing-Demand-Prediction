import os,sys
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from utils import load_data

train_set='../data/train.csv'
test_set='../data/test.csv'
target_path='../result/result.csv'
def Rmse(list1,list2):
    len_list1=len(list1)
    len_list2=len(list2)
    if not len_list1==len_list2:
        return 0
    sum=0
    for i in range(len_list1):
        sum+=(list1[i]-list2[i])**2
    result=math.sqrt(sum/len_list1)
    return result

if __name__=='__main__':

    labels_list,train_list=load_data()
    label_list=labels_list[:,1]
    len_list=len(label_list)
    indexs=np.array([[index for index in range(len_list) if train_list[index,0]==num] for num in range(6)])
    test_list=load_data(path=test_set,flag=1)
    len_test=len(test_list)
    test_indexs=np.array([[index for index in range(len_test) if test_list[index,0]==num] for num in range(6)])
    for i in range(6):
        if i==0:
            all_test_indices=test_indexs[0,]
        else:
            all_test_indices+=test_indexs[i,]
    all_test_indices=np.array(all_test_indices)
    ALL_mse=[]
    for i in range(6):
        indice=indexs[i,]
        sublabel_list=label_list[indice]
        subtrain_list=train_list[indice]
        test_indice=test_indexs[i,]
        subtest_list=test_list[test_indice]
        len_list=len(sublabel_list)
        split_list=KFold(len_list,n_folds=4,shuffle=True,random_state=False)
        run_time=0
        Rmses_rate=[]

        for train,test in split_list:
            x_train,x_test,y_train,y_test=train_list[train],train_list[test],label_list[train],label_list[test]
            global regr_rf
            regr_rf=RandomForestRegressor(max_depth=14,random_state=2)
            regr_rf.fit(x_train,y_train)
            pred=regr_rf.predict(x_test)
            Rmse_rate=Rmse(pred,y_test)
            Rmses_rate.append(Rmse_rate)
        #print np.mean(Rmses_rate)
        ALL_mse.append(Rmses_rate)
        if i==0:
            pred_test=regr_rf.predict(subtest_list)
        else:
            pred_test=np.concatenate([pred_test,regr_rf.predict(subtest_list)],axis=0)
        result_output=np.zeros(len_test)

    for i in range(len_test):
        result_output[i]=pred_test[all_test_indices==i][0]
    time_info=pd.read_csv('../data/sampleSubmission.csv')
    time_info=np.array(time_info['datetime'])
    df=pd.DataFrame({'datetime':time_info,'count':result_output})
    df.to_csv(target_path,columns=['datetime','count'],index=False)
    print 'mean mse of 6 folds: ',np.mean(ALL_mse)

