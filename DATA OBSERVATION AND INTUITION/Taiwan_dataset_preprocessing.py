# Taiwan Data preprocessing
import pandas as pd
import numpy as np
import sys
import os

def get_data():
    cause_id_96 = "11"  #  ICD-9 code
    cause_id_97 = "09"

    for year in range(81, 109):
        dic = {}
        file = open("./DATA/Taiwan/dead" + str(year) + ".txt")
        while True:
            line = file.readline()
            if line:
                # print(line)
                line = line.strip().split(",")
                # print(line)
                country = line[1]
                cause = line[2]
                number = line[5]
                # print(cause)
                if year <= 96 and str(cause) == cause_id_96:
                    # print(number)
                    if country in dic:
                        dic[country] = dic[country] + int(number)
                    else:
                        dic[country] = int(number)
                elif year > 96 and str(cause) == cause_id_97:
                    # print(number)
                    if country in dic:
                        dic[country] = dic[country] + int(number)
                    else:
                        dic[country] = int(number)
            else:
                break
            # sys.exit(1)
        print(dic)
        break

if __name__ == '__main__':
    print()
    cause_id_96 = "11"  #ICD-9
    cause_id_97 = "09"
    disease_name = "Diabetes"
    df_id = pd.read_csv("./DATA/Taiwan/ID.csv", encoding='utf-8')
    print(df_id)
    country_list = list(df_id['county'])
    name_list = list(df_id['name'])
    dic_name_id = {}
    for i in range(len(country_list)):
        dic_name_id[country_list[i]] = name_list[i]

    #get population data
    df_population = pd.read_csv("./DATA/Taiwan/population.csv")
    print(df_population)

    df_res = pd.DataFrame()
    for year in range(100, 109):
        dic = {}
        file = open("./DATA/Taiwan/dead" + str(year) + ".txt")
        while True:
            line = file.readline()
            if line:
                # print(line)
                line = line.strip().split(",")
                # print(line)
                country = line[1]
                cause = line[2]
                number = line[5]
                # print(cause)
                if year <= 96 and str(cause) == cause_id_96:
                    # print(number)
                    if country in dic:
                        dic[country] = dic[country] + int(number)
                    else:
                        dic[country] = int(number)
                elif year > 96 and str(cause) == cause_id_97:
                    # print(number)
                    if country in dic:
                        dic[country] = dic[country] + int(number)
                    else:
                        dic[country] = int(number)
            else:
                break
            # sys.exit(1)
        print(dic)

        population_region = list(df_population[str("region")])
        population_list = list(df_population[str(year)])

        res_id = []
        res_rate = []
        count = 0
        for item in dic:
            res_id.append(str(item))
            print(item)
            print(dic_name_id[int(item)])
            print(dic[item])

            region1 = dic_name_id[int(item)][0:3]
            region2 = dic_name_id[int(item)][3:]

            print(region1,region2)
            index1 = population_region.index(region1)
            print(index1)
            for i in range(len(population_region)):
                if population_region[i] == region2:
                    print(region2)
                    print("i=",i)
                    if i>index1:
                        break
            pp = population_list[i]
            while "," in pp:
                pp = pp.replace(",","")

            rate = dic[item]/int(pp)
            print(rate)
            print()
            res_rate.append(round(rate,4))
        print(res_id)
        print(res_rate)
        if year == 100:
            df_res['region'] = res_id
            df_res[disease_name+"_"+str(year)] = res_rate
            df_res = df_res.set_index('region')
        else:
            df_temp = pd.DataFrame()
            df_temp['region'] = res_id
            df_temp[disease_name + "_"+ str(year)] = res_rate
            df_temp = df_temp.set_index('region')
            df_res = pd.merge(df_res, df_temp, on='region')
        #break
    df_res.to_csv("./DATA/Taiwan_"+disease_name+".csv")