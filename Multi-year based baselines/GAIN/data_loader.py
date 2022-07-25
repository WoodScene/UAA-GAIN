'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import os
import numpy as np
import pandas as pd
from utils import binary_sampler
from keras.datasets import mnist
import random
import sys

disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']
disease_chinese = ['冠心病', '中风', '高血压', '糖尿病', '肺疾病', '癫痫',
                      '癌症', '心理健康', '哮喘', '心力衰竭', '姑息治疗', '痴呆',
                      '抑郁', '慢性肾病', '心房颤动', '肥胖', '学习障碍', '动脉疾病', ]


def data_loader(data_name, sampling_rate, year, disease_id, begin_year):
  seed = 25
  random.seed(seed + year)

  disease_select_list = disease_id
  N1 = 624
  N3 = year - 2008 + 1
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')

  for y in range(2008,year+1):
    df = pd.read_csv("../../DATA/UK_Obesity.csv")
    ward_code_list=list(df['Ward Code'])
    # print(list(df))
    df = df[disease_list[disease_select_list]+"_"+str(y)]
    data_x[:,y - 2008] = df.values

  miss_data_x = data_x.copy()

  ward_number = int(N1 * sampling_rate)  # 挑选格子的数目
  print("ward_number:",str(ward_number))
  for y in range(N3 - 1, N3):
    # print(y)
    data_year = data_x[:,y]

    # 随机挑选几百个格子
    ward_list = []  # 记录被挑选了的格子的id
    ward_nor_list = []  # 没有被挑选格子的id，用于计算测试误差;
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

  #处理m矩阵历史年份
  for idx_m in range((begin_year-2008),N3-1):
    #print("laile laodi")
    seed = 25
    random.seed(seed + idx_m + 2008)
    # 随机挑选几百个格子
    ward_list_m = []  # 记录被挑选了的格子的id
    ward_nor_list_m = []  # 没有被挑选格子的id，用于计算测试误差;
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
      data_m[i, idx_m] = 0
    #print("ward_list_m", sorted(ward_list_m))


  #print(len(np.where(data_m == 0)[0]))
  #sys.exit(1)
  miss_data_x[data_m == 0] = np.nan

  # print(data_x[:,-1])
  # print(miss_data_x[:, -1])
  # print(data_m[:, -1])
  return data_x, miss_data_x, data_m, ward_nor_list,ward_list,ward_code_list