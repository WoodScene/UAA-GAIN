#baseline
#NMF: For each disease, use Non-negative Matrix Factorization
#to predict the missing values.

import numpy
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def matrix_factorization(R,P,Q,K,epsilon,beta):
    Q=Q.T
    matrix_temp = numpy.dot(P, Q)
    position_nor_0 = numpy.where((R) != 0)
    original = R[position_nor_0]
    result = matrix_temp[position_nor_0]
    loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
    # print(loss_t1)
    loss_t1 = loss_t1 / len(original)
    loss_t = loss_t1 + epsilon + 1

    t0 = 100
    t=t0;
    while abs(loss_t-loss_t1)>epsilon:
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-numpy.dot(P[i,:],Q[:,j])
                    if R[i][j]>0:
                        alpha = 1 / sqrt(t)
                        t+=1
                        P[i,:]=P[i,:]+alpha*(2*eij*Q[:,j]-beta*P[i,:])
                        Q[:,j]=Q[:,j]+alpha*(2*eij*P[i,:]-beta*Q[:,j])
        loss_t = loss_t1
        matrix_temp=numpy.dot(P,Q)
        position_nor_0 = numpy.where((R) != 0)
        original = R[position_nor_0]
        result = matrix_temp[position_nor_0]
        loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
        # print(loss_t1)
        loss_t1 = loss_t1 / len(original)
    return P,Q.T

def get_matrix_R(sampling_rate,year,disease_id):
    seed = 25
    random.seed(seed + year)

    disease_select_list = disease_id[0]
    N1 = 624
    N3 = year - 2008 + 1
    data_x = numpy.ones((N1, N3), dtype='float64')

    for y in range(2008, year + 1):

        df = pd.read_csv("../../DATA/UK_Obesity.csv")
        ward_code_list = list(df['Ward Code'])
        # print(list(df))
        df = df[disease_list[disease_select_list] + "_" + str(y)]
        data_x[:, y - 2008] = df.values


    ward_number = int(N1 * sampling_rate / 100)
    print("ward_number:", str(ward_number))
    for y in range(N3 - 1, N3):
        # print(y)
        data_year = data_x[:, y]

        ward_list = []  #
        ward_nor_list = []
        num = 0

        while num < ward_number:
            id = random.randint(0, N1 - 1)
            if id in ward_list:
                continue
            ward_list.append(id)
            num = num + 1

        for i in range(N1):
            if i in ward_list:
                continue
            ward_nor_list.append(i)
            data_x[i,-1] = 0

        print("ward_list", sorted(ward_list))
        print("len ward_list", len(ward_list))
        print("len ward_nor_list", len(ward_nor_list))

    for idx_m in range((begin_year - 2008), N3 - 1):
        # print("laile laodi")
        seed = 25
        random.seed(seed + idx_m + 2008)

        ward_list_m = []
        ward_nor_list_m = []
        num = 0

        while num < ward_number:
            id = random.randint(0, N1 - 1)
            if id in ward_list_m:
                continue
            ward_list_m.append(id)
            num = num + 1

        for i in range(N1):
            if i in ward_list_m:
                continue
            ward_nor_list_m.append(i)
            data_x[i, idx_m] = 0
        # print("ward_list_m", sorted(ward_list_m))


    return data_x,ward_nor_list, ward_code_list, ward_list


def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):
    print(""+disease_list[diease_id]+" results：")
    #print(R_result)
    year=yy
    df_diease = pd.read_csv("../../DATA/UK_Obesity.csv")
    n=0
    y_rmse=0
    y_mae=0
    y_mape=0
    aaa = df_diease[disease_list[diease_id]+"_" + str(year)]
    R_original = numpy.ones((1,dimention), dtype='float64')
    R_original=aaa.values
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]
        if str(origial)!="nan":
            #print(origial[rate], result[rate])
            y_rmse=y_rmse+pow((origial-result),2)
            y_mae = y_mae + abs(origial - result)
            if origial==0:
                y_mape = y_mape + 1
            else:
                y_mape = y_mape + (abs(origial - result) / origial)
            n+=1

    RMSE=sqrt(y_rmse/n)
    MAE=y_mae/n
    MAPE = y_mape / n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    print("MAPE:", MAPE)
    return RMSE, MAE, MAPE


def MF(select_diease_id,sampling_rate,year):
    print(year)
    file_name = "NMF_Obesity_begin_year_" + str(begin_year) \
                + "_sampling_year_" + str(year) + "_sampling_rate_" + str(sampling_rate) + ".csv"

    years = ["2008-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')

    RES = []


    R,ward_nor_list, ward_code_list, ward_list = get_matrix_R(sampling_rate,year,select_diease_id)

    N=len(R)    #R rows
    M=len(R[0]) #R cols
    K=5
    P=numpy.random.uniform(0, 1, (N,K)) #
    Q=numpy.random.uniform(0, 1, (M,K)) #
    epsilon = 0.001
    beta = 0.0001
    #print(P)
    nP,nQ=matrix_factorization(R,P,Q,K,epsilon,beta)
    #print(R)
    R_MF=numpy.dot(nP,nQ.T)
    #print(R_MF)

    R_original = R[:, -1]
    R_result = R_MF[:, -1]
    for i in ward_list:
      R_result[i] = R_original[i]
    RMSE, MAE, MAPE = test_loss(R_result, ward_nor_list, year, select_diease_id[0], 624)


    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)

    df[disease_list[select_diease_id[0]]] = RES
    df.to_csv("./result/" + file_name)

if __name__=='__main__':
    sampling_rate = 10
    begin_year = 2008
    select_diease_id = [15]

    for year in range(begin_year,2018):


        MF(select_diease_id,sampling_rate,year)