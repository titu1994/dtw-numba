import numpy as np
import time

from utils.data_loader import load_dataset
from dtw import dtw_distance as numba_distance
from odtw import dtw_distance as numba_dataset_distance

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()


X0 = X_train[[0]]
X1 = X_test[[0]]

COUNT = 1000

# Compile first
_ = numba_distance(X0[0], X1[0])
_ = numba_dataset_distance(X0[0:1], X1[0:1])

baseline_t1 = time.time()
for i in range(COUNT):
    sam_distance = numba_distance(X0[0], X1[0])

baseline_t2 = time.time()

print("Numba time in ms : ", (baseline_t2 - baseline_t1) / float(COUNT))

numba_t1 = time.time()
for i in range(COUNT):
    som_distance = numba_dataset_distance(X0[[0]], X1[[0]])[0, 0]

numba_t2 = time.time()

print('Cuda time in ms : ', (numba_t2 - numba_t1) / float(COUNT))


print('Sam dist : ', sam_distance)
print('Som dist : ', som_distance)

print("MSE : ", np.square(sam_distance - som_distance))


