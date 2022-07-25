import numpy as np
import heapq
import copy
var_matrix = np.random.rand(10)
print(var_matrix)

var_temp = [1,34,5,1,43,234,23,45,3,2]
var_matrix = var_temp
print(var_matrix)
var_matrix = list(var_matrix)
aa =[1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]
min_num_index=map(var_matrix.index, heapq.nsmallest(3,var_matrix))
print(list(min_num_index))


tmp_list=copy.deepcopy(var_matrix)
tmp_list.sort()
max_num_index=[var_matrix.index(one) for one in tmp_list[::-1][:3]]
min_num_index=[var_matrix.index(one) for one in tmp_list[:3]]
print(list(min_num_index))


data = np.array([[1,2,1],[4,4,1]])

data_m = np.array([[0,0,1],[0,1,0]])
var_matrix = np.random.rand(2, 3)
print(var_matrix)
temp = var_matrix*(1-data_m)
print(temp)
res = []
for i in range(2):
    row = list(temp[i,:])
    print(row)
    while 0 in row:
        row.remove(0)
    print(row)
    mean = sum(row)/len(row)
    res.append(mean)
print(res)