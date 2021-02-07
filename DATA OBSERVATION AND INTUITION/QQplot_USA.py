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
    df = pd.read_csv("./DATA/USA_500city.csv")
    for year in range(2017, 2020):

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
        plt.hist(x2)
        plt.grid(alpha=0.5, linestyle='-.')  #
        plt.title(year)
        #plt.show()
        plt.savefig("./image/" + "USA_Histogram.png", dpi=600, bbox_inches='tight')
        break

def QQplot(diease_id):
    print()
    df = pd.read_csv("./DATA/USA_500city.csv")
    for year in range(2016, 2020):

        col_list = [diease_id + "_" + str(year)]
        df2 = df[col_list]
        # print(df2)
        delete1 = df2[df2[col_list[0]] == 0].index.tolist()
        df2 = df2.drop(delete1)
        print(df2)
        x = list(df2[diease_id + "_" + str(year)])
        print(x)
        x2 = [math.log(xx) for xx in x]
        fig = plt.figure()
        res = stats.probplot(x2, plot=plt)
        #plt.show()

        #plt.figure()
        plt.grid(alpha=0.5, linestyle='-.')
        plt.savefig("./image/" + "USA_QQplot.png", dpi=600, bbox_inches='tight')
        #plt.show()
        break

def KS_test(diease_id):
    print()
    df = pd.read_csv("./DATA/USA_50city.csv")
    for year in range(1999, 2018):
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
    df = pd.read_csv("./DATA/USA_500city.csv")
    for year in range(2016, 2020):
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
    diease_id = "Diabetes"
    Histogram(diease_id)
    QQplot(diease_id)
    #KS_test(diease_id)
    #SW_test(diease_id)