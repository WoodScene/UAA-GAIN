'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from data_loader import data_loader
from gain import gain
from utils import rmse_loss
from math import *
import random
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def test_loss(ori_data_x,imputed_data_x,ward_nor_list):#计算测试误差
    n=0
    y_rmse=0
    y_mae=0
    y_mape=0

    R_original=ori_data_x[:,-1]
    R_result = imputed_data_x[:,-1]
    yy_mae = []
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]
        #print(id,origial,result)

        if str(origial)!="nan" and origial!=0:
            #print(origial, result)
            y_rmse=y_rmse+pow((origial-result),2)
            y_mae = y_mae + abs(origial - result)
            yy_mae.append(abs(origial - result))
            y_mape = y_mape + (abs(origial - result) / origial)
            n += 1
    RMSE=sqrt(y_rmse/n)
    MAE=y_mae/n
    MAPE = y_mape / n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    print("MAPE:", MAPE)
    print()
    return RMSE, MAE, MAPE


def main (args, year, disease_id, begin_year, times):

    '''Main function for UCI letter and spam datasets.

    Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

    Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
    '''

    data_name = args.data_name
    sampling_rate = args.sampling_rate

    gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
    # gain_parameters = {'batch_size': 128,
    #                    'hint_rate': 0.9,
    #                    'alpha': 100,
    #                    'iterations': 10000}
    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m, ward_nor_list, ward_list, ward_code_list = data_loader(data_name, sampling_rate, year, disease_id, begin_year)

    # Impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters,times)

    RMSE, MAE, MAPE = test_loss(ori_data_x,imputed_data_x,ward_nor_list)

    R_original = ori_data_x[:, -1]
    R_result = imputed_data_x[:, -1]
    for i in ward_list:
      R_result[i] = R_original[i]

    return RMSE, MAE, MAPE, ward_code_list, R_result

if __name__ == '__main__':
    #data:08-17
    sampling_rate = 10
    begin_year = 2008
    disease_id = 15

    for year in range(2008,2018):
        print("sampling year:",year)

        file_name = "GAIN_Obesity_begin_year_"+ str(begin_year) \
                    + "_sampling_year_"+ str(year) + "_sampling_rate_"+ str(sampling_rate) +".csv"

        years=["2008-"+str(i) for i in range(year,year+1)]
        df=pd.DataFrame()
        df['year']=[val for val in years for i in range(3)]
        df=df.set_index('year')

        RES = []

        # Inputs for the main function
        parser = argparse.ArgumentParser()
        parser.add_argument(
          '--data_name',
          choices=['letter','spam'],
          default='spam',
          type=str)
        parser.add_argument(
          '--sampling_rate',
          help='sampling data probability',
          default=sampling_rate/100,
          type=float)
        parser.add_argument(
          '--batch_size',
          help='the number of samples in mini-batch',
          default=128,
          type=int)
        parser.add_argument(
          '--hint_rate',
          help='hint probability',
          default=0.9,
          type=float)
        parser.add_argument(
          '--alpha',
          help='hyperparameter',
          default=100,
          type=float)
        parser.add_argument(
          '--iterations',
          help='number of training interations',
          default=10000,
          type=int)

        args = parser.parse_args()
        print(args)
        #sys.exit(1)
        # Calls main function
        mmin = 100
        MAPE2 = 100
        RMSE_list = []
        MAE_list = []
        MAPE_list = []
        for count in range(5):
            print("No. of times", str(count+1))
            RMSE, MAE, MAPE, ward_code_list, R_result = main(args, year, disease_id, begin_year, count)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            MAPE_list.append(MAPE)
            if RMSE+MAE < mmin and MAPE2 > MAPE:
                RMSE2= RMSE
                MAE2 = MAE
                MAPE2 = MAPE
                mmin = RMSE+MAE

        RES.append(RMSE2)
        RES.append(MAE2)
        RES.append(MAPE2)
        RMSE_mean = np.mean(RMSE_list)
        MAE_mean = np.mean(MAE_list)
        MAPE_mean = np.mean(MAPE_list)

        RMSE_var = np.var(RMSE_list)
        MAE_var = np.var(MAE_list)
        MAPE_var = np.var(MAPE_list)

        df[disease_list[disease_id] + "_min"] = RES
        df[disease_list[disease_id] + "_mean"] = [RMSE_mean, MAE_mean, MAPE_mean]
        df[disease_list[disease_id] + "_var"] = [RMSE_var, MAE_var, MAPE_var]

        df['iterations'] = [str(args.iterations),"",""]
        df['hint_rate'] = [str(args.hint_rate),"",""]
        #df.to_csv("./result/" + file_name)

        print("final results:")
        print("RMSE:", RMSE_mean)
        print("MAE:", MAE_mean)
        print("MAPE:", MAPE_mean)