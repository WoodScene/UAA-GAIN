# Necessary packages
import heapq
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import random
import itertools
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def gain (data_x, gain_parameters, data_weight, ward_list, ward_nor_list):
  seed = 25
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)

  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']

  sample_times = 10
  # Other parameters
  no, dim = data_x.shape

  var_matrix = np.random.rand(no)
  # Hidden state dimensions
  h_dim = int(dim)
  #print(h_dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)


  X = tf.placeholder(tf.float32, shape = [None, dim])

  X_ALL = tf.placeholder(tf.float32, shape = [None, dim])

  #batch_size
  X2 = tf.placeholder(tf.float32, shape=[batch_size, dim])
  M2 = tf.placeholder(tf.float32, shape=[batch_size, dim])

  X3 = tf.placeholder(tf.float32, shape=[no, dim])
  M3 = tf.placeholder(tf.float32, shape=[no, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask weight vector
  M_weight = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*3, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  #G_W1 = tf.Variable(tf.random_normal([dim*2, h_dim], seed=225))
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h, var):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h, var], axis = 1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
  G_sample2 = G_sample
  G_sample_all = G_sample
  G_sample_one = G_sample
  G_sample3 = tf.constant(0, shape=[1, dim], dtype=tf.float32)
  G_sample4 = tf.constant(0, shape=[1, dim], dtype=tf.float32)
  Var = tf.constant(0, shape=[1, dim], dtype=tf.float32)
  Var2 = tf.constant(0, shape=[1, dim], dtype=tf.float32)
  Var3 = tf.constant(0, shape=[1, dim], dtype=tf.float32)


  for i in range(batch_size):
    #print(G_sample2[i*sample_times:(i+1)*sample_times,:])
    mean, var = tf.nn.moments(x=G_sample2[i*sample_times:(i+1)*sample_times,:], axes=[0])
    G_sample3 = tf.concat(values=[G_sample3, tf.reshape(mean,[1,dim])], axis=0)
    Var = tf.concat(values=[Var, tf.reshape(var, [1, dim])], axis=0)


  for i in range(no):
    #print(G_sample2[i*sample_times:(i+1)*sample_times,:])
    mean, var = tf.nn.moments(x=G_sample_all[i*sample_times:(i+1)*sample_times,:], axes=[0])
    Var2 = tf.concat(values=[Var2, tf.reshape(var, [1, dim])], axis=0)

  for i in range(batch_size):
    #print(G_sample2[i*sample_times:(i+1)*sample_times,:])
    mean, var = tf.nn.moments(x=G_sample_one[i*sample_times:(i+1)*sample_times,:], axes=[0])
    G_sample4 = tf.concat(values=[G_sample4, tf.reshape(mean,[1,dim])], axis=0)
    Var3 = tf.concat(values=[Var3, tf.reshape(var, [1, dim])], axis=0)


  G_sample3 = G_sample3[1:, :]
  G_sample4 = G_sample4[1:, :]
  Var = Var[1:, :]
  Var2 = Var2[1:, :]
  Var3 = Var3[1:, :]


  Var_weight = Var
  Row_max = tf.reduce_max(Var_weight, axis=1)
  Weight_matrix = tf.constant(0, shape=[1, dim], dtype=tf.float32)
  for i in range(batch_size):
    Weight_matrix = tf.concat(values=[Weight_matrix, tf.reshape(tf.div(Var_weight[i]+ 1e-8, Row_max[i]+ 1e-8), [1, dim])], axis=0)
  Weight_matrix = Weight_matrix[1:, :]

  N = sample_times  # number of repetition
  K = Weight_matrix.shape[0]  # for here 3

  order = list(range(0, N * K, K))
  order = [[x + i for x in order] for i in range(K)]
  order = list(itertools.chain.from_iterable(order))
  x_rep = tf.gather(tf.tile(Weight_matrix, [N, 1]), order)

  #Weight_matrix = 1e17 * M * x_rep
  #Mmin = tf.reduce_max(x_rep)
  #Weight_matrix = (1/(tf.reduce_max(x_rep) * 1e-1)) * M * x_rep
  Weight_matrix2 = M * x_rep

  #------------

  Var_last_col = Var2[:,-1]
  # Combine with observed data
  Hat_X = X2 * M2 + G_sample3 * (1 - M2)

  # Discriminator
  D_prob = discriminator(Hat_X, H, Var)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M2 * tf.log(D_prob + 1e-8) \
                                + (1-M2) * tf.log(1. - D_prob + 1e-8))
  
  G_loss_temp = -tf.reduce_mean((1-M2) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((Weight_matrix2 * X - Weight_matrix2 * G_sample)**2) / tf.reduce_mean(M)

  MSE_loss_init = \
    tf.reduce_mean((M_weight * X_ALL - M_weight * G_sample) ** 2) / tf.reduce_mean(M_weight)

  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss
  G_loss_init = alpha * MSE_loss_init
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  G_solver_init = tf.train.AdamOptimizer().minimize(G_loss_init, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for it in range(1000):
    X_all = norm_data_x[ward_list, :]

    X_mb = norm_data_x[ward_list, :]
    M_mb = data_m[ward_list, :]
    M_mb[:,-1] = 0
    M_ww = data_weight[ward_list, :]
    batch_size2 = len(ward_list)
    Z_mb = uniform_sampler(0, 1, batch_size2, dim)
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    # sys.exit(1)
    _, MSE_loss_curr = \
    sess.run([G_solver_init, G_loss_init],
             feed_dict = {X: X_mb, M: M_mb, M_weight: M_ww, X_ALL: X_all})
    if it % 100 == 0:
      print("pre-train mse loss:",MSE_loss_curr)

  for it in range(1000):
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    #print(len(batch_idx))
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]
    M_ww = data_weight[batch_idx, :]
    # print("z mb:",Z_mb)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
    M2_mb = np.repeat(M_mb, repeats=sample_times, axis=0)
    M2_ww = np.repeat(M_ww, repeats=sample_times, axis=0)
    X2_mb = np.repeat(X_mb, repeats=sample_times, axis=0)
    Z2_mb = uniform_sampler(0, 1, batch_size * sample_times, dim)
    X2_mb = M2_mb * X2_mb + (1 - M2_mb) * Z2_mb
    # Sample random vectors
    Z_mb = uniform_sampler(0, 1, batch_size, dim)
    #print(X_mb.shape)
    #print(M_mb.shape)
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                              feed_dict={M: M2_mb, X: X2_mb, H: H_mb, X2: X_mb, M2: M_mb})

  iii = 0
  # Start Iterations
  for it in tqdm(range(iterations)):
    # Sample batch

    # var_matrix = list(var_matrix)
    # print(var_matrix)
    # min_num_index = map(var_matrix.index, heapq.nsmallest(batch_size, var_matrix))
    # batch_idx = list(min_num_index)
    # print("batch_idx",batch_idx)
    #****************************
    var_matrix2 = list(var_matrix)
    #print("var_matrix2", sorted(var_matrix2))

    batch_idx = []
    ward_len = 0
    ward_max = int(batch_size/10)
    while len(batch_idx)<batch_size:
      min_id = var_matrix2.index(min(var_matrix2))
      if min_id not in ward_list:
        batch_idx.append(min_id)
      else:
        if ward_len < ward_max:
          ward_len = ward_len + 1
          batch_idx.append(min_id)
      var_matrix2[min_id] = 100
    #print("batch_idx", sorted(batch_idx))
    #print("batch_idx", batch_idx)


    #batch_idx = sample_batch_index(no, batch_size)

    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]
    M_ww = data_weight[batch_idx, :]

    #print("z mb:",Z_mb)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp

    M2_mb = np.repeat(M_mb, repeats=sample_times, axis=0)
    M2_ww = np.repeat(M_ww, repeats=sample_times, axis=0)
    X2_mb = np.repeat(X_mb, repeats=sample_times, axis=0)
    Z2_mb = uniform_sampler(0, 0.01, batch_size*sample_times, dim)
    X2_mb = M2_mb * X2_mb + (1 - M2_mb) * Z2_mb
    #print(X_mb)
    #print(X2_mb)
    #sys.exit(1)
    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 


    _, D_loss_curr, wt, var, xr = sess.run([D_solver, D_loss_temp, Weight_matrix, Var, x_rep],
                              feed_dict = {M: M2_mb, X: X2_mb, H: H_mb, X2:X_mb, M2:M_mb })

    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X2_mb, M: M2_mb, H: H_mb, X2:X_mb, M2:M_mb, M_weight: M2_ww})

    X_mb = norm_data_x
    M_mb = data_m
    M_ww = data_weight
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, no, dim)
    H_mb = M_mb * H_mb_temp

    M2_mb = np.repeat(M_mb, repeats=sample_times, axis=0)
    M2_ww = np.repeat(M_ww, repeats=sample_times, axis=0)
    X2_mb = np.repeat(X_mb, repeats=sample_times, axis=0)
    Z2_mb = uniform_sampler(0, 0.01, no*sample_times, dim)
    X2_mb = M2_mb * X2_mb + (1 - M2_mb) * Z2_mb

    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    var_list_temp = sess.run([Var_last_col],
                              feed_dict={M: M2_mb, X: X2_mb, H: H_mb})
    #print(var_matrix)
    var_matrix = var_list_temp[0]

    #print(var_list_temp)

    # if iii == 0:
    #   iii = 1
    #   print("MSE_loss_curr ",MSE_loss_curr)
    if it % int(iterations/20) == 0:
      print("MSE_loss_curr ",MSE_loss_curr)
  ## Return imputed data
  Z_mb = uniform_sampler(0, 0.01, no, dim)
  M_mb = data_m
  M_ww = data_weight
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)

  # MSE_loss_final = sess.run([MSE_loss],
  #          feed_dict={X: X_mb, M: M_mb, M_weight: M_ww})
  # print("MSE loss final:",MSE_loss_final).
  sess.close()
  return imputed_data