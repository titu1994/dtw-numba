import numpy as np
import time
from numba import cuda

from utils.data_loader import load_dataset
from dtw import slow_dtw_distance
from dtw import dtw_distance as numba_sample_distance
from odtw import dtw_distance as numba_dataset_distance
from cdtw import dtw_distance as cuda_dataset_distance

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()

X0 = X_train.reshape((X_train.shape[0], -1))
X1 = X_test.reshape((X_test.shape[0], -1))

COUNT = 1

# Compile first
_ = numba_sample_distance(X0[0], X1[0])
_ = numba_dataset_distance(X0[0:1], X1[0:1])

threads_per_block = (32, 32)
blocks_per_grid = (X0.shape[0] // threads_per_block[0] + 1,
                   X1.shape[0] // threads_per_block[1] + 1)


cuda.select_device(0)

distance_mat = np.zeros((X0.shape[0], X1.shape[0]), dtype=np.float64)
cuda_dataset_distance[blocks_per_grid, threads_per_block](X0, X1, distance_mat)

cuda.close()

print(dir(cuda.current_context()))
print("Memory info : ", cuda.current_context().get_memory_info())
cuda.current_context().deallocations.clear()
print("Memory info : ", cuda.current_context().get_memory_info())

# slow_t1 = time.time()
#
# slow_distance = np.empty((X0.shape[0], X1.shape[0]))
# for i in range(X0.shape[0]):
#     for j in range(X1.shape[0]):
#         slow_distance[i, j] = slow_dtw_distance(X0[i], X1[j])
#
# slow_t2 = time.time()

# print("Non Numba optimized time : ", (slow_t2 - slow_t1) / float(COUNT))

sample_t1 = time.time()

sample_distance = np.empty((X0.shape[0], X1.shape[0]))
for i in range(X0.shape[0]):
    for j in range(X1.shape[0]):
        sample_distance[i, j] = numba_sample_distance(X0[i], X1[j])

sample_t2 = time.time()

print("Sample optimized time : ", (sample_t2 - sample_t1) / float(COUNT))

dataset_t1 = time.time()

dataset_distance = numba_dataset_distance(X0, X1)

dataset_t2 = time.time()

print('Dataset optimized time : ', (dataset_t2 - dataset_t1) / float(COUNT))

dataset_t1 = time.time()

# cuda.select_device(0)
# cuda.synchronize()

distance_mat = np.empty((X0.shape[0], X1.shape[0]), dtype=np.float64)
cuda_dataset_distance[blocks_per_grid, threads_per_block](X0, X1, distance_mat)

# cuda.close()

dataset_t2 = time.time()

print("Cuda optimized time : ", (dataset_t2 - dataset_t1) / float(COUNT))
print()

# print('Non Optimized dist mean : ', slow_distance.mean())
print('Sample Optimized mean dist : ', sample_distance.mean())
print('Dataset Optimized mean dist : ', dataset_distance.mean())
print()

print("MSE  : ", np.mean(np.square(distance_mat - sample_distance)))
# print("MSE (non optimized - sample optimized): ", np.mean(np.square(slow_distance - sample_distance)))
# print("MSE (non optimized - dataset optimized): ", np.mean(np.square(slow_distance - dataset_distance)))


