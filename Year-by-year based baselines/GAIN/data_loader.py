# Necessary packages
import os
import numpy as np
import pandas as pd
from utils import binary_sampler
from keras.datasets import mnist
import random


disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def data_loader(data_name, sampling_rate, year, disease_id, begin_year):
  seed = 25
  random.seed(seed + year)

  disease_select_list = disease_id
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
      file_name = "GAIN_Obesity_begin_year_" + str(begin_year) \
                      + "_sampling_year_" + str(y) + "_sampling_rate_" + str(int(sampling_rate * 100)) + "_data.csv"
      df = pd.read_csv("./data/" + file_name)
      ward_code_list = list(df['ward code'])
      df = df[disease_list[disease_select_list]]
      data_x[:, y - 2008] = df.values

  miss_data_x = data_x.copy()

  ward_number = int(N1 * sampling_rate)
  print("ward_number:",str(ward_number))
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

  return data_x, miss_data_x, data_m, ward_nor_list,ward_list,ward_code_list