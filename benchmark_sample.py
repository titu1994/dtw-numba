import numpy as np
import time

from dtw import slow_dtw_distance
from dtw import dtw_distance as numba_sample_distance
from odtw import dtw_distance as numba_dataset_distance
from utils.data_loader import load_dataset

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()


X0 = X_train[[0]]
X1 = X_test[[0]]

COUNT = 100

# Compile first
_ = numba_sample_distance(X0[0], X1[0])
_ = numba_dataset_distance(X0[0:1], X1[0:1])

slow_t1 = time.time()
for i in range(COUNT):
    slow_distance = slow_dtw_distance(X0[0], X1[0])

slow_t2 = time.time()

print("Non Numba optimized time : ", (slow_t2 - slow_t1) / float(COUNT))

sample_t1 = time.time()
for i in range(COUNT):
    sample_distance = numba_sample_distance(X0[0], X1[0])

sample_t2 = time.time()

print("Sample optimized time : ", (sample_t2 - sample_t1) / float(COUNT))

dataset_t1 = time.time()
for i in range(COUNT):
    dataset_distance = numba_dataset_distance(X0[[0]], X1[[0]])[0, 0]

dataset_t2 = time.time()

print('Dataset optimized time : ', (dataset_t2 - dataset_t1) / float(COUNT))
print()

print('Non Optimized dist : ', slow_distance)
print('Numba Optimized dist : ', sample_distance)
print('Dataset Optimized dist : ', dataset_distance)

print()
print("MSE (non optimized - sample optimized): ", np.square(slow_distance - sample_distance))
print("MSE (non optimized - dataset optimized): ", np.square(slow_distance - dataset_distance))


