import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#disease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
disease_list = ['Coronary Heart Disease Prevalence', 'Stroke or Transient Ischaemic Attacks (TIA) Prevalence',
                   'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence',
                   'Chronic Obstructive Pulmonary Disease Prevalence', 'Epilepsy Prevalence', 'Cancer Prevalence',
                   'Mental Health Prevalence', 'Asthma Prevalence', 'Heart Failure Prevalence',
                   'Palliative Care Prevalence', 'Dementia Prevalence', 'Depression Prevalence',
                   'Chronic Kidney Disease Prevalence', 'Atrial Fibrillation Prevalence', 'Obesity Prevalence',
                   'Learning Disabilities Prevalence', 'Cardiovascular Disease Primary Prevention Prevalence']

def test_loss(R_result, ward_nor_list, yy, diease_id, dimention):
    print(""+disease_list[diease_id]+" results：")
    #print(R_result)
    year=yy
    df_diease = pd.read_csv("../../DATA/UK_Obesity.csv")
    n=0
    y_rmse=0
    y_mae=0
    y_mape=0
    aaa = df_diease[disease_list[diease_id]+"_" + str(year)]
    R_original = np.ones((1,dimention), dtype='float64')
    R_original=aaa.values
    for id in ward_nor_list:
        result=R_result[id]
        origial=R_original[id]
        if str(origial)!="nan" and origial!=0:
            #print(origial[rate], result[rate])
            y_rmse=y_rmse+pow((origial-result),2)
            y_mae = y_mae + abs(origial - result)
            y_mape = y_mape + (abs(origial - result) / origial)
            n+=1

    RMSE=sqrt(y_rmse/n)
    MAE=y_mae/n
    MAPE = y_mape / n
    print("RMSE:",RMSE)
    print("MAE:",MAE)
    print("MAPE:", MAPE)
    return RMSE, MAE, MAPE

def get_tensor_A(sampling_rate, year, select_diease_id):
    seed = 25
    random.seed(seed + year)
    N1 = 624
    N2 = len(list(select_diease_id))
    N3 = year - 2008 + 1

    R = np.ones((N3, N1, N2), dtype='float64')
    ward_code_list=[]

    for y in range(2008, year + 1):
        if y < begin_year or y == year:
            df = pd.read_csv("../../DATA/UK_Obesity.csv")
            ward_code_list = list(df['Ward Code'])
            # print(list(df))
            df = df[disease_list[select_diease_id[0]] + "_" + str(y)]
            R[y - 2008, :, 0] = df.values

        else:
            file_name = "AutoEncoder_Obesity_begin_year_" + str(begin_year) \
                        + "_sampling_year_" + str(y) + "_sampling_rate_" + str(int(sampling_rate)) + "_data.csv"
            df = pd.read_csv("./data/" + file_name)
            ward_code_list = list(df['ward code'])
            df = df[disease_list[select_diease_id[0]]]
            R[y - 2008, :, 0] = df.values

    for i in range(0, len(R)):
        for j in range(0, len(R[0])):
            for k in range(0, len(R[0][0])):
                if np.isnan(R[i][j][k]):
                    R[i,j,k] = 0

    R_original = R
    ward_number = int(N1 * sampling_rate /100)

    for y in range(N3 - 1, N3):
        data_year = R[y]
        ward_list = []  #
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
            R[y, i, 0] = 0
        print("ward_list", sorted(ward_list))
        print("len ward_list", len(ward_list))
        print("len ward_nor_list", len(ward_nor_list))
    return R, ward_nor_list, ward_code_list, ward_list

if __name__=='__main__':
    sampling_rate = 10
    begin_year = 2008 #
    select_diease_id = [15]

    for year in range(2008,2018):
        print("sampling year:",year)

        file_name = "AutoEncoder_Obesity_begin_year_"+ str(begin_year) \
                    + "_sampling_year_"+ str(year) + "_sampling_rate_"+ str(sampling_rate) +".csv"

        years=["2008-"+str(i) for i in range(year,year+1)]
        df=pd.DataFrame()
        df['year']=[val for val in years for i in range(3)]
        df=df.set_index('year')

        RES = []
        # Calls main function
        mmin = 100
        MAPE2 = 100
        RMSE_list = []
        MAE_list = []
        MAPE_list = []
        for count in range(5):
            seed = 25 + count
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            A, ward_nor_list, ward_code_list, ward_list=get_tensor_A(sampling_rate,year,select_diease_id)
            input_dimension=len(A[0])
            #print(A.shape)
            R_original = A[-1,: , 0]
            #print("original",R_original)
            #sys.exit(0)
            id_0=[]
            input_x=np.array([[i for i in range(input_dimension)]])
            for j in range(len(A)):
                for k in range(len(select_diease_id)):
                    #print()
                    A1=A[j,:,k]
                    A2=np.array(A1).reshape((1,input_dimension))
                    input_x=np.vstack((input_x,A2))
                    id=np.where((A2)!=0)
                    for i in range(len(id[0])):
                        temp=[]
                        temp.append(list(id[0])[i]+j)
                        temp.append(list(id[1])[i])
                        id_0.append(temp)
            input_x=input_x[1:,:]
            input_pre=np.array([[i for i in range(input_dimension)]])
            for k in range(len(select_diease_id)):
                # print()
                A1 = A[len(A)-1, :, k]
                A2 = np.array(A1).reshape((1, input_dimension))
                input_pre = np.vstack((input_pre, A2))
            input_pre = input_pre[1:, :]
            print("len id0", len(id_0))
            # Parameter
            learning_rate = 0.0005
            training_epochs = 5 #
            batch_size = 256
            display_step = 1
            times = 15000
            # Network Parameters
            n_input = input_dimension
            dim_x=32
            dim_y=32

            X=tf.placeholder("float",[None,n_input])
            #X=tf.placeholder("float",[None,n_input])

            # hidden layer settings
            n_hidden_1 = 32 # 1st layer num features
            n_hidden_2 = 32 # 2nd layer num features
            weights = {
                    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=225+count)),
                'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],  seed=225+count)),
                'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],seed=225+count)),
                'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input], seed=225+count)),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], seed=225+count)),
                'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], seed=225+count)),
                'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1], seed=225+count)),
                'decoder_b2': tf.Variable(tf.random_normal([n_input], seed=225+count)),
                }

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h1'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['encoder_h2'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h1'])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights['decoder_h2'])

            # Building the encoder
            def encoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                               biases['encoder_b1']))

                layer_1 = layer_1 + layer_1

                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                               biases['encoder_b2']))
                return layer_2

            # Building the decoder
            def decoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                               biases['decoder_b1']))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                               biases['decoder_b2']))
                return layer_2

            # Construct model
            encoder_op = encoder(X) 			# 128 Features
            decoder_op = decoder(encoder_op)	# 784 Features

            # Prediction
            y_pred = decoder_op	# After
            # Targets (Labels) are the input data.
            y_true = X			# Before

            def loss_cal(y_true,y_pred):
                y_true2 = tf.gather_nd(y_true,id_0)
                y_pred2 = tf.gather_nd(y_pred,id_0)
                return tf.pow(y_true2 - y_pred2, 2)

            # Define loss and optimizer, minimize the squared error
            cost = tf.reduce_mean(loss_cal(y_true,y_pred))

            regularizer = tf.contrib.layers.l2_regularizer(0.01)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            cost += reg_term

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            # Launch the graph
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(times):
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: input_x})
                    if i%1000==0:
                        print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(c))
                print("Optimization Finished!")
                # # Applying encode and decode over test set
                A_result = sess.run(
                    y_pred, feed_dict={X: input_pre})
                #print(A_result.shape)
                for dd in range(len(select_diease_id)):
                    diease_id=select_diease_id[dd]
                    RMSE, MAE, MAPE=test_loss(A_result[dd,:], ward_nor_list, year, diease_id, input_dimension)
                    R_result = A_result[dd,:]
                    continue

            for i in ward_list:
                R_result[i] = R_original[i]

            RMSE_list.append(RMSE)
            MAE_list.append(MAE)
            MAPE_list.append(MAPE)
            #sys.exit(1)
            if RMSE+MAE < mmin and MAPE2 > MAPE:
                RMSE2= RMSE
                MAE2 = MAE
                MAPE2 = MAPE
                R_result2 = R_result
                mmin = RMSE+MAE

        df_csv = pd.DataFrame()
        file_name_csv = "AutoEncoder_Obesity_begin_year_" + str(begin_year) \
                        + "_sampling_year_" + str(year) + "_sampling_rate_" + str(sampling_rate) + "_data.csv"
        df_csv['ward code'] = ward_code_list
        df_csv[disease_list[select_diease_id[0]]] = R_result2
        df_csv.to_csv("./data/" + file_name_csv, encoding='utf_8_sig')

        RES.append(RMSE2)
        RES.append(MAE2)
        RES.append(MAPE2)
        RMSE_mean = np.mean(RMSE_list)
        MAE_mean = np.mean(MAE_list)
        MAPE_mean = np.mean(MAPE_list)

        RMSE_var = np.var(RMSE_list)
        MAE_var = np.var(MAE_list)
        MAPE_var = np.var(MAPE_list)
        df[disease_list[select_diease_id[0]] + "_min"] = RES
        df[disease_list[select_diease_id[0]] + "_mean"] = [RMSE_mean, MAE_mean, MAPE_mean]
        df[disease_list[select_diease_id[0]] + "_var"] = [RMSE_var, MAE_var, MAPE_var]

        df.to_csv("./result/" + file_name)
        print("final results:")
        print("RMSE:", RMSE_mean)
        print("MAE:", MAE_mean)
        print("MAPE:", MAPE_mean)
