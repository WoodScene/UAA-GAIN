# Pearson
# KL
# JS
# Wessertein Distance
import scipy.stats
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

#  pearson
def Mining_pearson():
    diease_id = 3
    df = pd.read_csv("./DATA/UK_Obesity.csv")
    corr_list = []
    x = []
    for length in range(1,10):
        res = []
        x.append(length)
        for year in range(2008,2018):
            year2 = year + length
            if year2 > 2017:
                break
            years = [year, year2]
            col_list = [disease_list[diease_id] + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            #print(df2)
            delete1 = df2[df2[col_list[0]]==0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            #print(df2)

            corr_value = df2.corr()
            # print(corr_value)
            res.append(abs(round(list(corr_value[disease_list[diease_id] + "_" + str(years[1])])[0], 4)))
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x,corr_list)
    plt.grid(alpha=0.5, linestyle='-.')

    plt.xlabel("Time interval",fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel("Pearson value",fontdict={'weight': 'normal', 'size': 20})
    #plt.show()
    plt.savefig("./image/" + "UK_pearson.png",dpi=600, bbox_inches='tight')

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

def Minging_JS(diease_id):

    df = pd.read_csv("./DATA/UK_Obesity.csv")
    corr_list = []
    x = []
    for length in range(1, 10):
        res = []
        x.append(length)
        for year in range(2008, 2018):
            year2 = year + length
            if year2 > 2017:
                break
            years = [year, year2]
            col_list = [disease_list[diease_id] + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            # print(df2)
            p = np.array(list(df2[col_list[0]]))
            q = np.array(list(df2[col_list[1]]))
            #print(q,p)
            js = JS_divergence(p, q)
            # print(corr_value)
            res.append(js)
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x, corr_list)
    plt.grid(alpha=0.5, linestyle='-.')

    plt.xlabel("Time interval",fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel("JS divergence",fontdict={'weight': 'normal', 'size': 20})
    #plt.show()
    plt.savefig("./image/" + "UK_JS.png",dpi=600, bbox_inches='tight')

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def Minging_KL(diease_id):
    df = pd.read_csv("./DATA/UK_Obesity.csv")
    corr_list = []
    x = []
    for length in range(1, 10):
        res = []
        x.append(length)
        for year in range(2008, 2018):
            year2 = year + length
            if year2 > 2017:
                break
            years = [year, year2]
            col_list = [disease_list[diease_id] + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            # print(df2)
            p = np.array(list(df2[col_list[0]]))
            q = np.array(list(df2[col_list[1]]))
            #print(q,p)
            js = KL_divergence(p, q)
            # print(corr_value)
            res.append(js)
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x, corr_list)
    plt.title("KL")
    plt.xlabel("Time span",fontdict={'weight': 'normal', 'size': 20})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

def Minging_Wasserstein(diease_id):
    print("Wasserstein")
    df = pd.read_csv("./DATA/UK_Obesity.csv")
    corr_list = []
    x = []
    for length in range(1, 10):
        res = []
        x.append(length)
        for year in range(2008, 2018):
            year2 = year + length
            if year2 > 2017:
                break
            years = [year, year2]
            col_list = [disease_list[diease_id] + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            # print(df2)
            p = np.array(list(df2[col_list[0]]))
            q = np.array(list(df2[col_list[1]]))
            #print(q,p)
            wd = wasserstein_distance(p,q)
            # print(corr_value)
            res.append(wd)
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x, corr_list)
    plt.title("Wasserstein")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

if __name__ == '__main__':
    diease_id = 15
    Mining_pearson()
    Minging_JS(diease_id)
    #Minging_KL(diease_id)
    #Minging_Wasserstein(diease_id)