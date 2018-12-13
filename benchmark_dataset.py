import numpy as np
import time

from utils.data_loader import load_dataset
from dtw import dtw_distance as numba_distance
from odtw import dtw_distance as numba_dataset_distance

X_train, y_train, X_test, y_test = load_dataset('adiac', normalize_timeseries=True)
print()

X0 = X_train.reshape((X_train.shape[0], -1))
X1 = X_test.reshape((X_test.shape[0], -1))

COUNT = 1

# Compile first
_ = numba_distance(X0[0], X1[0])
_ = numba_dataset_distance(X0[0:1], X1[0:1])

baseline_t1 = time.time()

sam_distance = np.empty((X0.shape[0], X1.shape[0]))
for i in range(X0.shape[0]):
    for j in range(X1.shape[0]):
        sam_distance[i, j] = numba_distance(X0[i], X1[j])

baseline_t2 = time.time()

print("Numba time in ms : ", (baseline_t2 - baseline_t1) / float(COUNT))

numba_t1 = time.time()

som_distance = numba_dataset_distance(X0, X1)

numba_t2 = time.time()

print('Numba dataset time in ms : ', (numba_t2 - numba_t1) / float(COUNT))


print('Sam mean dist : ', sam_distance.mean())
print('Som mean dist : ', som_distance.mean())

print("MSE : ", np.mean(np.square(sam_distance - som_distance)))


