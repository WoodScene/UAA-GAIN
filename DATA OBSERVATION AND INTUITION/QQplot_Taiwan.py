
#Verify that the data for each year follows a normal distribution

import scipy.stats
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats
from scipy.stats import shapiro
import math

def Histogram(diease_id):
    print()
    df = pd.read_csv("./DATA/Taiwan_" + diease_id + "_norm.csv")
    for year in range(108, 109):
        col_list = [diease_id + "_" + str(year)]
        df2 = df[col_list]
        # print(df2)
        delete1 = df2[df2[col_list[0]] == 0].index.tolist()
        df2 = df2.drop(delete1)
        print(df2)
        x = list(df2[diease_id + "_" + str(year)])
        print(x)
        plt.figure()
        x2 = [math.log(xx) for xx in x]
        plt.hist(x)
        plt.grid(alpha=0.5, linestyle='-.')  #
        plt.title("2019")
        plt.savefig("./image/" + "TW_Histogram.png", dpi=600, bbox_inches='tight')
        #plt.show()
        break

def QQplot(diease_id):
    print()
    df = pd.read_csv("./DATA/Taiwan_" + diease_id + ".csv")
    for year in range(107, 109):
        col_list = [diease_id + "_" + str(year)]
        df2 = df[col_list]
        # print(df2)
        delete1 = df2[df2[col_list[0]] == 0].index.tolist()
        df2 = df2.drop(delete1)
        print(df2)
        x = list(df2[diease_id + "_" + str(year)])
        print(x)

        fig = plt.figure()
        x2 = [math.log(xx) for xx in x]
        res = stats.probplot(x2, plot=plt)
        #plt.show()
        #plt.figure()  # 初始化一张图
        plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
        plt.savefig("./image/" + "TW_QQplot.png", dpi=600, bbox_inches='tight')
        #plt.show()
        break

def KS_test(diease_id):
    print()
    df = pd.read_csv("./DATA/Taiwan_" + diease_id + ".csv")
    for year in range(100, 109):

        col_list = [diease_id + "_" + str(year)]
        df2 = df[col_list]
        # print(df2)
        delete1 = df2[df2[col_list[0]] == 0].index.tolist()
        df2 = df2.drop(delete1)
        print(df2)
        x = list(df2[diease_id + "_" + str(year)])
        print(x)
        print(kstest(x, cdf="norm"))
        break

def SW_test(diease_id):
    #print()
    df = pd.read_csv("./DATA/Taiwan_" + diease_id + ".csv")
    for year in range(100, 109):
        print(year)
        col_list = [diease_id + "_" + str(year)]
        df2 = df[col_list]
        # print(df2)
        delete1 = df2[df2[col_list[0]] == 0].index.tolist()
        df2 = df2.drop(delete1)
        #print(df2)
        x = list(df2[diease_id + "_" + str(year)])
        #print(x)
        x2 = [math.log(xx) for xx in x]
        print(shapiro(x2))

        #break

if __name__ == '__main__':
    diease_id =  "Diabetes"
    Histogram(diease_id)
    QQplot(diease_id)
    # KS_test(diease_id)
    SW_test(diease_id)