#baseline
#For each disease, take the temporal or spatial average
#as a complementary value.
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
#diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def get_tensor_A(sampling_rate, year, select_diease_id):
    seed = 25
    random.seed(seed + year)
    N1 = 624
    N2 = len(list(select_diease_id))
    N3 = year - 2008 + 1

    R = np.ones((N3, N1, N2), dtype='float64')  # ?x624x18
    ward_code_list=[]

    for y in range(2008, year + 1):

        df = pd.read_csv("../../DATA/UK_Obesity.csv")
        ward_code_list = list(df['Ward Code'])
        # print(list(df))
        df = df[disease_list[select_diease_id[0]] + "_" + str(y)]
        R[y - 2008, :, 0] = df.values

    for i in range(0, len(R)):
        for j in range(0, len(R[0])):
            for k in range(0, len(R[0][0])):
                if np.isnan(R[i][j][k]):
                    R[i,j,k] = 0

    R_original = R
    ward_number = int(N1 * sampling_rate /100)

    for y in range(N3 - 1, N3):
        data_year = R[y]
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
            R[y, i, 0] = 0
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
            R[idx_m, i, 0] = 0
        # print("ward_list_m", sorted(ward_list_m))

    return R, ward_nor_list, ward_code_list, ward_list

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
    R_original = np.ones((1,dimention), dtype='float64')
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


def Time_mean(select_diease_id,sampling_rate,year):
    print(year)
    file_name = "Mean_time_Obesity_begin_year_" + str(begin_year) \
                + "_sampling_year_" + str(year) + "_sampling_rate_" + str(sampling_rate) + ".csv"

    years = ["2008-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')

    RES = []
    print("Mean_time")

    A, ward_nor_list, ward_code_list, ward_list = get_tensor_A(sampling_rate, year, select_diease_id)
    A_result = [0 for i in range(624)]
    R_original = A[-1, :, 0]
    for i in ward_nor_list:
        time_list = A[:, i, 0]
        time_list = list(time_list[0:len(time_list)-1])
        #print(time_list)
        while 0 in time_list:
            time_list.remove(0)
        if len(time_list)==0:
            #print("cont")
            ave_all = [A[-1, i, 0] for i in ward_list]
            mean = np.median(ave_all)
            A_result[i] = mean
            continue

        mean = np.median(time_list)
        A_result[i] = mean

    # sys.exit(1)
    for i in ward_list:
        A_result[i] = R_original[i]
    #sys.exit(1)
    RMSE, MAE, MAPE = test_loss(A_result, ward_nor_list, year, select_diease_id[0], 624)

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

        Time_mean(select_diease_id,sampling_rate,year)