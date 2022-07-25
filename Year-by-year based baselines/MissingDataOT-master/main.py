
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from math import *
import random
import os
import sys
from geomloss import SamplesLoss

from imputers import OTimputer, RRimputer

from utils import *
from data_loaders import dataset_loader
from softimpute import softimpute, cv_softimpute

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

torch.set_default_tensor_type('torch.DoubleTensor')
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def data_loader2(sampling_rate, year, disease_id, begin_year):
  seed = 25
  random.seed(seed + year)
  disease_select_list = disease_id
  #year = 2017
  N1 = 624
  N3 = year - 2008 + 1
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')


  for y in range(2008,year+1):
    if y < begin_year or y == year:
      df = pd.read_csv("../../DATA/UK_Obesity.csv")
      ward_code_list=list(df['Ward Code'])
      # print(list(df))
      df = df[disease_list[disease_select_list]+"_"+str(y)]
      data_x[:,y - 2008] = df.values

    else:
      file_name = "Optimal_transport_Obesity_begin_year_" + str(begin_year) \
                      + "_sampling_year_" + str(y) + "_sampling_rate_" + str(int(sampling_rate)) + "_data.csv"
      df = pd.read_csv("./data2/" + file_name)
      ward_code_list = list(df['ward code'])
      df = df[disease_list[disease_select_list]]
      data_x[:, y - 2008] = df.values

  miss_data_x = data_x.copy()


  ward_number = int(N1 * sampling_rate /100)

  for y in range(N3 - 1, N3):
    # print(y)
    data_year = data_x[:,y]

    ward_list = []
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
      data_m[i,-1] = 0
    print("ward_list", sorted(ward_list))
    print("len ward_list", len(ward_list))
    print("len ward_nor_list", len(ward_nor_list))

  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, ward_nor_list,ward_list,ward_code_list

def test_loss(ori_data_x,imputed_data_x,ward_nor_list):
    n = 0
    y_rmse = 0
    y_mae = 0
    y_mape = 0

    R_original = ori_data_x[:, -1]
    R_result = imputed_data_x[:, -1]
    yy_mae = []
    for id in ward_nor_list:
        result = R_result[id]
        origial = R_original[id]
        # print(id,origial,result)
        if str(origial) != "nan" and origial != 0:
            # print(origial, result)
            y_rmse = y_rmse + pow((origial - result), 2)
            y_mae = y_mae + abs(origial - result)
            yy_mae.append(abs(origial - result))
            y_mape = y_mape + (abs(origial - result) / origial)
            n += 1

    RMSE = sqrt(y_rmse / n)
    MAE = y_mae / n
    MAPE = y_mape / n
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("MAPE:", MAPE)
    print()
    return RMSE, MAE, MAPE


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':

    setup_seed(25)
    if not os.path.isdir("datasets"):
        os.mkdir("datasets")

    sampling_rate = 10
    begin_year = 2008
    disease_id = 15
    for year in range(2008,2018):
        file_name = "Optimal_transport_Obesity_begin_year_" + str(begin_year) \
                    + "_sampling_year_" + str(year) + "_sampling_rate_" + str(sampling_rate) + ".csv"

        years = ["2008-" + str(i) for i in range(year, year + 1)]
        df = pd.DataFrame()
        df['year'] = [val for val in years for i in range(3)]
        df = df.set_index('year')
        RES = []
        mmin = 100
        MAPE2 = 100
        RMSE_list = []
        MAE_list = []
        MAPE_list = []
        for count in range(5):
            setup_seed(25 + count)
            ground_truth, miss_data_x, ward_nor_list,ward_list,ward_code_list = data_loader2(sampling_rate, year, disease_id, begin_year)
            R_original = ground_truth[:, -1]

            X_true = torch.from_numpy(ground_truth)

            print(X_true.shape)

            X_miss = torch.from_numpy(miss_data_x)
            #print(X_miss)

            n, d = X_miss.shape
            batchsize = 128 # If the batch size is larger than half the dataset's size,
                            # it will be redefined in the imputation methods.
            #0.01
            lr = 0.01

            print(lr)
            epsilon = pick_epsilon(X_miss) # Set the regularization parameter as a multiple of the median distance, as per the paper.
            print(epsilon)

            sk_imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=2000)

            sk_imp, sk_maes, sk_rmses = sk_imputer.fit_transform(X_miss, verbose=True, report_interval=500, X_true=X_true)

            X_true = X_true.numpy()
            sk_imp = sk_imp.detach().numpy()
            R_result = sk_imp[:, -1]
            for i in ward_list:
                R_result[i] = R_original[i]
            RMSE, MAE, MAPE = test_loss(ground_truth, sk_imp, ward_nor_list)
            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            MAPE_list.append(MAPE)
            if RMSE + MAE < mmin and MAPE2 > MAPE:
                RMSE2 = RMSE
                MAE2 = MAE
                MAPE2 = MAPE
                R_result2 = R_result
                mmin = RMSE + MAE

        df_csv = pd.DataFrame()
        file_name_csv = "Optimal_transport_Obesity_begin_year_" + str(begin_year) \
                        + "_sampling_year_" + str(year) + "_sampling_rate_" + str(sampling_rate) + "_data.csv"
        df_csv['ward code'] = ward_code_list
        df_csv[disease_list[disease_id]] = R_result2
        df_csv.to_csv("./data2/" + file_name_csv, encoding='utf_8_sig')

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
        df['RMSE_list'] = [str(RMSE_list),"",""]
        df['MAPE_list'] = [str(MAPE_list),"",""]
        df.to_csv("./result/" + file_name)
        print("final results:")
        print("RMSE:", RMSE_mean)
        print("MAE:", MAE_mean)
        print("MAPE:", MAPE_mean)