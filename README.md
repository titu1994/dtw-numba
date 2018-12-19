# Dynamic Time Warping in Python using Numba

Implementation of Dynamic Time Warping algorithm with speed improvements based on Numba.

Supports for K nearest neighbours classifier using Dynamic Time Warping, based on the [work presented by Mark Regan](https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping). The classes called `KnnDTW` are obtained from there, as a simplified interface akin to Scikit-Learn.

Thanks to [Sam Harford](https://github.com/sharford5) for providing the core of the DTW computation.

Thanks to [Jonas Klemming](https://github.com/klon) for the C implementation of `ucrdtw`.


## Dynamic Time Warping Variants
-----
The three variants available are in `dtw.py`, `odtw.py` and `ucrdtw.py`.

- `dtw.py`: Single threaded variant, support for visualizing the progress bar.
- `odtw.py`: Multi threaded variant, no support for visualization. In practice, much more effiecient.
- `ucrdtw.py`: **Experimental (Do not use)**. Multi threaded variant, no support for visualization. It is based upon the optimized C implementation available at https://github.com/klon/ucrdtw.

`odtw.py` is further optimized to run on entire datasets in parallel, and therefore is preferred for any task involving classification.

`ucrdtw.py` is a highly efficient alternative to `odtw.py`, which provides the ability to select warping window and online z-normalization of the dataset. Currently, it is not as performant as the optimized C version, and the original codebase should be used instead.

 - **NOTE**: Due to an inefficient implementation, the `ucrdtw` implementation is much slower than `odtw` for certain datasets. However, for warping window less than 100%, it often surpasses `odtw`. To keep scores from evaluations equivalent, all evaluated results will be done with an infinite warping window (the entire length of the query series).

## Speed optimizations
-----
While [Numba](http://numba.pydata.org/) supports pure python code as input to be compiled, it benefits from C-level micro-optimizations. Considering the runtime complexity of DTW, the `dtw_distance` method in `odtw.py` is a more efficient DTW computation implementation in Numba, which disregards python syntax for C-level optimizations.

Some optimizations shared by both include : 

- Empty allocation of `E` : avoids filling with 0s.
- Inlined `max` operation to avoid depending on python `max` function.

Optimizations available to `odtw.py` : 

- Remove calls to np.square() and compute difference and square manually to avoid additional function calls.
- Parallelize computation of distance matrix over two datasets.

Optimizations available to `ucrdtw.py` :

- Computation of lower bounds to exit DTW computation early
- Compiled data structures and operations on Deque for fast computation of lower bounds
- Compiled reverse sorting via Quicksort, falling back to reverse insertion sort for small subsets.
- Compiled ops for minimum, maximum, absolute value and square distance
- Cached allocations compared to the C version
- Choice for online z-normalization
- Caching of sorted query-index pair for faster evaluation of Index results in parallel

# Evaluations against UCR Archive
To ensure that the performance of the two DTW models is exactly the same as that of the DTW scores available in the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), provided is the `Adiac` dataset, which is loaded, z-normalized, then used for evaluation. All three DTW implementations obtain same scores at 100% warping window.

```
Test Accuracy : 0.6035805626598465
Test Error : 0.3964194373401535
```

These scores match those in the above repository for DTW (w=100).

# Speed Comparison
Comparisons were made against an Intel i7-6700HQ CPU @ 2.60 GHz (8 Logical CPUs, 4 Physical CPUs), with 16 GB of RAM on an Alienware R2 (2015) laptop. Tests were performed on the Adiac dataset, which contains 390 train samples, 391 test samples and each sample is a univariate time series of length 176 timesteps.

## Sample level test
Here, we compare the time taken to compute the DTW distance between the first train and test samples of the Adiac dataset. 

Output : 
```
Non Numba Optimized time :  0.12019050598144532
Sample optimized time :  8.00013542175293e-05
Dataset optimized time :  0.0003000330924987793
UCR optimized time :  0.0005000114440917968

Non Optimized dist :  1.1218082709896633
Sample Optimized dist :  1.1218082709896633
Dataset Optimized dist :  1.1218082709896633
UCR Optimized dist :  1.1218082709896633

MSE (non optimized - sample optimized):  0.0
MSE (non optimized - dataset optimized):  0.0
MSE (non optimized - ucr optimized):  0.0
```

Key observations are : 

- Non-Numba optimized code is several orders of magnitude slower than the sample or dataset optimized variants.
- Dataset optimized method is slightly slower than the sample variant. This is because the cost incurred with initializing and running subprocesses for a single sample is greater than the parallelization benefit of the underlying optimizations.
- MSE between the non optimized variant and sample or dataset optimized variant is 0. Therefore this speed does not come at the cost of accuracy.

## Dataset level test
Here, we compute the time taken to compute the DTW distance matrix between the entire train set (390, 176) against the entire test set (391, 176). This yields a distance matrix of shape [390, 391].

Output : 
```
Non Numba Optimized time :  8386.9442625578
UCR optimized time :  8.214709758758545
Sample optimized time :  13.303221225738525
Dataset optimized time :  3.0960452556610107

Non Optimized dist mean :  0.9556927603445304
Sample Optimized mean dist :  0.9556927603445304
Dataset Optimized mean dist :  0.9556927603445304
UCR Optimized mean dist :  0.9556927652664071

MSE (non optimized - sample optimized):  0.0
MSE (non optimized - dataset optimized):  0.0
MSE (dataset optimized - ucr optimized):  1.2700471157257482e-15

```

## Summary

| Time in seconds | Non Optimized | Sample Optimized | UCR Optimized | Dataset Optimized |
|-----------------|:-------------:|:----------------:|:-------------:|:-----------------:|
| Single Sample   |    0.12019    |     8.01e-05     |    5.00e-4    |      3.00e-4      |
| Full Dataset    |   > 30 mins   |      13.3015     |     8.2147    |       3.096       |

Key observations are : 

- Non-Numba optimized code is several orders of magnitude slower than both of the optimized variants, so much so that it is not feasible.
- Dataset optimized method is several times faster than the sample optimized variant. Scaling is sub-linear, considering that an optimal scaled version should take 1/8-th the time of the sample variant, however it is still benefitial for longer time series (or larger dataset).
- MSE between the non optimized variant and the sample or dataset optimized variants is 0 once again.

# Requirements

- Numba (use `pip install numba` or `conda install numba`)
- Numpy
- Scipy
- Scikit-learn
- Pandas (to load UCR datasets)
- joblib (to extract UCR datasets)
