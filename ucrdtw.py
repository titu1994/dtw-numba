"""
This is a Numba optimized re-implementation of the UCR Suite
available at https://github.com/klon/ucrdtw/blob/master/src/ucrdtw.c
"""

import numpy as np
from numba import jit, prange, jitclass, float64, int64
from scipy.stats import mode
from sklearn.metrics import accuracy_score

__all__ = ['dtw_distance', 'KnnDTW']

INF = 1e9
EPSILON = 1e-8


@jit(nopython=True, parallel=True, nogil=True)
def dtw_distance(dataset1, dataset2, r, normalize):
    """
    Computes the dataset DTW distance matrix using multiprocessing.

    Args:
        dataset1: timeseries dataset of shape [N1, T1]
        dataset2: timeseries dataset of shape [N2, T2]
        r: warping window. If in between 0-1, considered
            as percentage of the query length. If larger
            than 0, and less than query length, used
            directly. If < 0, assumed to be full query
            length.
        normalize: Whether to perform online z-normalization.

    Returns:
        Distance matrix of shape [N1, N2]
    """
    n1 = dataset1.shape[0]
    n2 = dataset2.shape[0]
    l2 = dataset2.shape[1]
    dist = np.empty((n1, n2), dtype=np.float64)

    for j in prange(n2):
        # Precompute and cache the index array for j-th query sample
        q_temp = []
        for k in range(l2):
            idx = Index(dataset2[j][k], k)
            q_temp.append(idx)

        # sort using reverse quicksort
        quicksort(q_temp, 0, l2 - 1)

        for i in prange(n1):
            dist[i][j] = _dtw_distance(dataset1[i], dataset2[j], q_temp, r, normalize)

    return dist


@jit(nopython=True)
def dist(x, y):
    z = x - y
    return z * z


@jit(nopython=True)
def minimum(x, y):
    if x < y:
        return x
    else:
        return y


@jit(nopython=True)
def maximum(x, y):
    if x > y:
        return x
    else:
        return y


@jit(nopython=True)
def absolute_val(x):
    if x > 0:
        return x
    else:
        return -x


@jit(nopython=True)
def _reverse_partition(arr, low, high):
    pivot = low + (high - low) // 2

    temp = arr[high]
    arr[high] = arr[pivot]
    arr[pivot] = temp

    pivot_val = absolute_val(arr[high].value)

    i = low
    for j in range(low, high):
        if (absolute_val(arr[j].value) > pivot_val):
            temp1 = arr[j]
            arr[j] = arr[i]
            arr[i] = temp1
            i += 1

    # swapping pivot with its correct position
    temp2 = arr[high]
    arr[high] = arr[i]
    arr[i] = temp2

    return i


@jit(nopython=True)
def quicksort(arr, low, high):
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = _reverse_partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)


@jitclass([('value', float64),
           ('index', int64)])
class Index(object):

    def __init__(self, value, index):
        self.value = value
        self.index = index


@jitclass([('dq', int64[:]),
           ('size', int64),
           ('capacity', int64),
           ('f', int64),
           ('r', int64)])
class Deque(object):

    def __init__(self, capacity):
        self.dq = np.empty(capacity, dtype=np.int64)
        self.size = 0
        self.capacity = capacity
        self.f = 0
        self.r = capacity - 1


# Insert to the queue at the back
@jit(nopython=True)
def deque_push_back(deque, v):
    deque.dq[deque.r] = v
    deque.r -= 1

    if deque.r < 0:
        deque.r = deque.capacity - 1

    deque.size += 1


# Delete the current (front) element from queue
@jit(nopython=True)
def deque_pop_front(deque):
    deque.f -= 1

    if deque.f < 0:
        deque.f = deque.capacity - 1

    deque.size -= 1


# Delete the last element from queue
@jit(nopython=True)
def deque_pop_back(deque):
    deque.r = (deque.r + 1) % deque.capacity
    deque.size -= 1


# Get the value at the current position of the circular queue
@jit(nopython=True)
def deque_front(deque):
    aux = deque.f - 1

    if aux < 0:
        aux = deque.capacity - 1

    return deque.dq[aux]


@jit(nopython=True)
def deque_back(deque):
    aux = (deque.r + 1) % deque.capacity
    return deque.dq[aux]


@jit(nopython=True)
def deque_empty(deque):
    return deque.size == 0


"""
Finding the envelope of min and max value for LB_Keogh
Implementation idea is introduced by Danial Lemire in his paper
"Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
"""


@jit(nopython=True)
def lower_upper_lemire(t, len, r, l, u):
    du = Deque(2 * r + 2)
    dl = Deque(2 * r + 2)

    deque_push_back(du, 0.)
    deque_push_back(dl, 0.)

    for i in range(1, len):
        if i > r:
            u[i - r - 1] = t[deque_front(du)]
            l[i - r - 1] = t[deque_front(dl)]

        if t[i] > t[i - 1]:
            deque_pop_back(du)
            while ((not deque_empty(du)) and (t[i] > t[deque_back(du)])):
                deque_pop_back(du)

        else:
            deque_pop_back(dl)
            while ((not deque_empty(dl)) and (t[i] < t[deque_back(dl)])):
                deque_pop_back(dl)

        deque_push_back(du, i)
        deque_push_back(dl, i)

        if i == (2 * r + 1 + deque_front(du)):
            deque_pop_front(du)
        elif i == (2 * r + 1 + deque_front(dl)):
            deque_pop_front(dl)

    for i in range(len, len + r + 1):
        u[i - r - 1] = t[deque_front(du)]
        l[i - r - 1] = t[deque_front(dl)]

        if (i - deque_front(du) >= 2 * r + 1):
            deque_pop_front(du)

        if (i - deque_front(dl) >= 2 * r + 1):
            deque_pop_front(dl)


"""
Calculate quick lower bound
Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
However, because of z-normalization the top and bottom cannot give significant benefits.
And using the first and last points can be computed in constant time.
The pruning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
"""


@jit(nopython=True)
def lb_kim_hierarchy(t, q, j, len, mean, std, best_so_far):

    x0 = (t[j] - mean) / std
    y0 = (t[(len - 1 + j)] - mean) / std

    lb = dist(x0, q[0]) + dist(y0, q[len - 1])

    if (lb >= best_so_far):
        return lb

    if len < 2:
        return lb

    # 2 points at front
    x1 = (t[(j + 1)] - mean) / std
    d = minimum(dist(x1, q[0]), dist(x0, q[1]))
    d = minimum(d, dist(x1, q[1]))
    lb += d

    if (lb >= best_so_far):
        return lb

    # 2 points at back
    y1 = (t[(len - 2 + j)] - mean) / std
    d = minimum(dist(y1, q[len - 1]), dist(y0, q[len - 2]))
    d = minimum(d, dist(y1, q[len - 2]))
    lb += d

    if (lb >= best_so_far):
        return lb

    if len < 3:
        return lb

    # 3 points at front
    x2 = (t[(j + 2)] - mean) / std
    d = minimum(dist(x0, q[2]), dist(x1, q[2]))
    d = minimum(d, dist(x2, q[2]))
    d = minimum(d, dist(x2, q[1]))
    d = minimum(d, dist(x2, q[0]))
    lb += d

    if (lb >= best_so_far):
        return lb

    # 3 points at back
    y2 = (t[(len - 3 + j)] - mean) / std
    d = minimum(dist(y0, q[len - 3]), dist(y1, q[len - 3]))
    d = minimum(d, dist(y2, q[len - 3]))
    d = minimum(d, dist(y2, q[len - 2]))
    d = minimum(d, dist(y2, q[len - 1]))
    lb += d

    return lb


"""
// LB_Keogh 1: Create Envelope for the query
/// Note that because the query is known, envelope can be created once at the beginning.
///
/// Variable Explanation,
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
/// j     : index of the starting location in t
/// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
"""


@jit(nopython=True)
def lb_keogh_cumulative(order, t, uo, lo, cb, j, len, mean, std, best_so_far):
    lb = 0.0

    for i in range(len):
        if lb < best_so_far:
            x = (t[(order[i] + j)] - mean) / std
            d = 0.0

            if (x > uo[i]):
                v = x - uo[i]
                d = v * v

            elif (x < lo[i]):
                v = x - lo[i]
                d = v * v

            lb += d
            cb[order[i]] = d

        else:
            break

    return lb


"""
/// LB_Keogh 2: Create Envelop for the data
/// Note that the envelops have been created (in main function) when each data point has been read.
///
/// Variable Explanation,
/// tz: Z-normalized data
/// qo: sorted query
/// cb: (output) current bound at each position. Used later for early abandoning in DTW.
/// l,u: lower and upper envelope of the current data
"""


@jit(nopython=True)
def lb_keogh_data_cumulative(order, tz, qo, cb, l, u, len, mean, std, best_so_far):
    lb = 0.0

    for i in range(len):
        if lb < best_so_far:
            uu = (u[order[i]] - mean) / std
            ll = (l[order[i]] - mean) / std
            d = 0

            if (qo[i] > uu):
                d = dist(qo[i], uu)

            elif (qo[i] < ll):
                d = dist(qo[i], ll)

            lb += d
            cb[order[i]] = d

    return lb


@jit(nopython=True)
def dtw(A, B, cb, m, r, best_so_far, cost, cost_prev):
    for k in range(2 * r + 1):
        cost[k] = INF
        cost_prev[k] = INF

    for i in range(m):
        k = maximum(0, r - i)
        min_cost = INF

        j = maximum(0, i - r)
        while j <= min(m - 1, i + r):
            # Initialize all row and column
            if ((i == 0) and (j == 0)):
                cost[k] = dist(A[0], B[0])
                min_cost = cost[k]

                j += 1
                k += 1
                continue

            if ((j - 1 < 0) or (k - 1 < 0)):
                y = INF
            else:
                y = cost[k - 1]

            if ((i - 1 < 0) or (k + 1 > 2 * r)):
                x = INF
            else:
                x = cost_prev[k + 1]

            if ((i - 1 < 0) or (j - 1 < 0)):
                z = INF
            else:
                z = cost_prev[k]

            # Classic DTW calculation
            if x < y and x < z:
                val = x
            elif y < x and y < z:
                val = y
            else:
                val = z

            cost[k] = val + dist(A[i], B[j])

            # Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if (cost[k] < min_cost):
                min_cost = cost[k]

            j += 1
            k += 1

        # We can abandon early if the current cummulative distance with lower bound together are larger than best_so_far
        if (((i + r) < (m - 1)) and ((min_cost + cb[i + r + 1]) >= best_so_far)):
            return min_cost + cb[i + r + 1]

        # Move current array to previous array
        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp

    k -= 1

    # the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    return cost_prev[k]


@jit(nopython=True)
def _dtw_distance(series1, series2, q_temp, r, normalize):
    """
    Returns the DTW similarity distance between two 1-D
    timeseries numpy arrays.

    Args:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared.
        q_temp: The sorted sequence-index pair, precomputed
            for efficiency.
        r: warping window. If in between 0-1, considered
            as percentage of the query length. If larger
            than 0, and less than query length, used
            directly. If < 0, assumed to be full query
            length.
        normalize: Whether to perform online z-normalization.

    Returns:
        DTW distance between A and B
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]

    # compute window length
    if r > 0.0 and r < 1.0:
        r = int(r * l2)
    elif r < 0:
        r = l2
    
    # r must be a minimum of 1
    r = max(1, int(r))

    EPOCH = maximum(l1, l2)  # 100000

    kim = 0
    keogh = 0
    keogh2 = 0

    # Allocations
    # q = _series_copy(series2)
    q = np.copy(series2)
    qo = np.empty(l2)
    uo = np.empty(l2)
    lo = np.empty(l2)
    order = np.empty(l2, dtype=np.int32)
    u = np.empty(l2)
    l = np.empty(l2)
    cb = np.empty(l2)
    cb1 = np.empty(l2)
    cb2 = np.empty(l2)
    # u_d = np.zeros(l2)
    # l_d = np.zeros(l2)
    t = np.empty(l2 * 2)
    tz = np.empty(l2)
    buffer = np.empty(EPOCH)
    u_buff = np.empty(EPOCH)
    l_buff = np.empty(EPOCH)

    # DTW cost matrices, cached to prevent multiple allocations
    cost = np.empty(2 * r + 1)
    cost_prev = np.empty(2 * r + 1)

    best_so_far = INF

    # Read query
    if normalize:
        ex = ex2 = 0.

        for i in range(l2):
            d = q[i]
            ex += d
            ex2 += d * d

        # z-normalize the query, keep in same array, q
        mean = ex / float(l2)
        std = ex2 / float(l2)
        std = np.sqrt(std - mean * mean) + EPSILON

        for i in range(l2):
            q[i] = (q[i] - mean) / std

    # Create envelope of the query: lower envelope, l, and upper envelope, u
    lower_upper_lemire(q, l2, r, l, u)

    # also create another arrays for keeping sorted envelope
    for i in range(l2):
        o = q_temp[i].index
        order[i] = o
        qo[i] = q[o]
        uo[i] = u[o]
        lo[i] = l[o]

    i = 0  # current index of the data in current chunk of size EPOCH
    j = 0  # the starting index of the data in the circular array, t

    done = False
    it = 0
    data_index = 0

    while not done:
        # Read first m-1 points
        # ep = 0
        if it == 0:
            for k in range(l2 - 1):
                if k < (l2 - 1) and data_index < l1:
                    buffer[k] = series1[data_index]
                    data_index += 1

        else:
            for k in range(l2 - 1):
                buffer[k] = buffer[EPOCH - l2 + 1 + k]

        # Read buffer of size EPOCH or when all data has been read.
        ep = l2 - 1
        while (ep < EPOCH and data_index < l1):
            buffer[ep] = series1[data_index]
            data_index += 1
            ep += 1

        # Data are read in chunk of size EPOCH.
        # When there is nothing to read, the loop is end

        if ep <= l2 - 1:
            done = True
        else:
            lower_upper_lemire(buffer, ep, r, l_buff, u_buff)

            # Do main task here.
            ex = 0
            ex2 = 0

            for i in range(ep):
                # A bunch of data has been read and pick one of them at a time to use
                d = buffer[i]

                # Calcualte sum and sum square
                ex += d
                ex2 += d * d

                # t is a circular array for keeping current data
                t[i % l2] = d

                # Double the size for avoiding using modulo "%" operator
                t[(i % l2) + l2] = d

                # Start the task when there are more than m-1 points in the current chunk
                if i >= (l2 - 1):
                    mean = ex / l2
                    std = ex2 / l2
                    std = np.sqrt(std - mean * mean) + EPSILON

                    # compute the start location of the data in the current circular array, t
                    j = (i + 1) % l2

                    # the start location of the data in the current chunk
                    I = i - (l2 - 1)

                    # Use a constant lower bound to prune the obvious subsequence
                    lb_kim = lb_kim_hierarchy(t, q, j, l2, mean, std, best_so_far)

                    if (lb_kim < best_so_far):
                        # Use a linear time lower bound to prune; z_normalization of t will be computed on the fly
                        # uo, lo are envelope of the query.
                        lb_k = lb_keogh_cumulative(order, t, uo, lo, cb1, j, l2, mean, std, best_so_far)

                        if (lb_k < best_so_far):
                            # Take another linear time to compute z_normalization of t.
                            # Note that for better optimization, this can merge to the previous function

                            for k in range(l2):
                                tz[k] = (t[(k + j)] - mean) / std

                            # Use another lb_keogh to prune
                            # qo is the sorted query. tz is unsorted z_normalized data
                            # l_buff, u_buff are big envelope for all data in this chunk

                            lb_k2 = lb_keogh_data_cumulative(order, tz, qo, cb2, l_buff + I, u_buff + I, l2, mean, std,
                                                             best_so_far)

                            if lb_k2 < best_so_far:
                                # Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                                # Note that cb and cb2 will be cumulative summed here.
                                if (lb_k > lb_k2):
                                    cb[l2 - 1] = cb1[l2 - 1]

                                    for k in range(l2 - 2, -1, -1):
                                        cb[k] = cb[k + 1] + cb1[k]

                                else:
                                    cb[l2 - 1] = cb2[l2 - 1]

                                    for k in range(l2 - 2, -1, -1):
                                        cb[k] = cb[k + 1] + cb2[k]

                                # Compute DTW and early abandoning if possible
                                dist = dtw(tz, q, cb, l2, r, best_so_far, cost, cost_prev)

                                if dist < best_so_far:
                                    # Update best_so_far
                                    # loc is the real starting location of the nearest neighbor in the file
                                    best_so_far = dist

                            else:
                                keogh2 += 1

                        else:
                            keogh += 1

                    else:
                        kim += 1

                    ex -= t[j]
                    ex2 -= t[j] * t[j]

                else:
                    pass

            # If the size of last chunk is less then EPOCH, then no more data and terminate
            if (ep < EPOCH):
                done = True
            else:
                it += 1

    return np.sqrt(best_so_far)


# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class KnnDTW(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for KNN

    window : int, optional (default = -1).
        Warping window. If in between 0-1, considered
        as percentage of the query length. If larger
        than 0, and less than query length, used
        directly. If < 0, assumed to be full query
        length.

    normalize : bool, optional (default = False)
        Whether to perform online z-normalization.
    """

    def __init__(self, n_neighbors=1, window=-1, normalize=False):
        self.n_neighbors = n_neighbors
        self.window = window
        self.normalize = normalize

    def fit(self, x, y):
        """Fit the model using x as training data and y as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        y : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = np.copy(x)
        self.y = np.copy(y)

    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        dm = dtw_distance(x, y, self.window, self.normalize)

        return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """
        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.y[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

    def evaluate(self, x, y):
        """
        Predict the class labels or probability estimates for
        the provided data and then evaluates the accuracy score.

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

          y : array of shape [n_samples]
              Array containing the labels of the testing dataset to be classified

        Returns
        -------
          1 floating point value representing the accuracy of the classifier
        """
        # Predict the labels and the probabilities
        pred_labels, pred_probas = self.predict(x)

        # Ensure labels are integers
        y = y.astype('int32')
        pred_labels = pred_labels.astype('int32')

        # Compute accuracy measure
        accuracy = accuracy_score(y, pred_labels)
        return accuracy

    def predict_proba(self, x):
        """Predict the class labels probability estimates for
        the provided data

        Arguments
        ---------
            x : array of shape [n_samples, n_timepoints]
                Array containing the testing data set to be classified

        Returns
        -------
            2 arrays representing:
                (1) the predicted class probabilities
                (2) the knn labels
        """
        np.random.seed(0)
        dm = self._dist_matrix(x, self.x)

        # Invert the distance matrix
        dm = -dm

        # Compute softmax probabilities
        dm_exp = np.exp(dm - dm.max())
        dm = dm_exp / np.sum(dm_exp, axis=-1, keepdims=True)

        classes = np.unique(self.y)
        class_dm = []

        # Partition distance matrix by class
        for i, cls in enumerate(classes):
            idx = np.argwhere(self.y == cls)[:, 0]
            cls_dm = dm[:, idx]  # [N_test, N_train_c]

            # Take maximum distance vector due to softmax probabilities
            cls_dm = np.max(cls_dm, axis=-1)  # [N_test,]

            class_dm.append([cls_dm])

        # Concatenate the classwise distance matrices and transpose
        class_dm = np.concatenate(class_dm, axis=0)  # [C, N_test]
        probabilities = class_dm.transpose()  # [N_test, C]

        knn_labels = np.argmax(probabilities, axis=-1)

        return probabilities, knn_labels
