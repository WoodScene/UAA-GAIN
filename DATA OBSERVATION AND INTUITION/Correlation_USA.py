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


def Mining_pearson(diease_id):
    df = pd.read_csv("./DATA/USA_500city.csv")
    corr_list = []
    x = []
    for length in range(1,4):
        res = []
        x.append(str(length))
        for year in range(2016,2020):
            year2 = year + length
            if year2 > 2019:
                break
            years = [year, year2]
            col_list = [diease_id + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            #print(df2)
            delete1 = df2[df2[col_list[0]]==0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            corr_value = df2.corr()
            # print(corr_value)
            res.append(abs(round(list(corr_value[diease_id + "_" + str(years[1])])[0], 4)))
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x,corr_list)
    plt.grid(alpha=0.5, linestyle='-.')
    plt.xlabel("Time interval",fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel("Pearson value",fontdict={'weight': 'normal', 'size': 20})
    #plt.show()
    plt.savefig("./image/" + "USA_pearson.png", dpi=600, bbox_inches='tight')

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

def Minging_JS(diease_id):
    df = pd.read_csv("./DATA/USA_500city.csv")
    corr_list = []
    x = []
    for length in range(1, 4):
        res = []
        x.append(str(length))
        for year in range(2016, 2020):
            year2 = year + length
            if year2 > 2019:
                break
            years = [year, year2]
            col_list = [diease_id + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)

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
    plt.savefig("./image/" + "USA_JS.png",dpi=600, bbox_inches='tight')

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def Minging_KL(diease_id):
    df = pd.read_csv("./DATA/USA_500city.csv")
    corr_list = []
    x = []
    for length in range(1, 4):
        res = []
        x.append(str(length))
        for year in range(2016, 2020):
            year2 = year + length
            if year2 > 2019:
                break
            years = [year, year2]
            col_list = [diease_id + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
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
    plt.xlabel("Time span", fontdict={'weight': 'normal', 'size': 20})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


def Minging_Wasserstein(diease_id):
    print("Wasserstein")
    df = pd.read_csv("./DATA/USA_500city.csv")
    corr_list = []
    x = []
    for length in range(1, 4):
        res = []
        x.append(length)
        for year in range(2016, 2020):
            year2 = year + length
            if year2 > 2019:
                break
            years = [year, year2]
            col_list = [diease_id + "_" + str(years[i]) for i in range(2)]
            df2 = df[col_list]
            # print(df2)
            delete1 = df2[df2[col_list[0]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            delete1 = df2[df2[col_list[1]] == 0].index.tolist()
            df2 = df2.drop(delete1)
            p = np.array(list(df2[col_list[0]]))
            q = np.array(list(df2[col_list[1]]))
            #print(q,p)
            wd = wasserstein_distance(p,q)
            res.append(wd)
        corr_list.append(np.mean(res))
    print(corr_list)
    plt.plot(x, corr_list)
    plt.title("Wasserstein")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

if __name__ == '__main__':
    disease_name = "Diabetes"
    Mining_pearson(disease_name)
    Minging_JS(disease_name)
    #Minging_KL(disease_name)
    # Minging_Wasserstein(disease_name)