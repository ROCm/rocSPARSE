/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_BENCHMARK_UTILS_H_
#define ROCSPARSE_BENCHMARK_UTILS_H_

#include <stdlib.h>

template <typename T>
inline int gen2DLaplacianUS(int ndim, int **rowptr, int **col, T **val)
{

    int n = ndim * ndim;
    int nnz_mat = n * 5 - ndim * 4;

    *rowptr = (int*) malloc((n+1)*sizeof(int));
    *col = (int*) malloc(nnz_mat*sizeof(int));
    *val = (T*) malloc(nnz_mat*sizeof(T));

    int nnz = 0;

    // Fill local arrays
    for (int i=0; i<ndim; ++i)
    {
        for (int j=0; j<ndim; ++j)
        {
            int idx = i*ndim+j;
            (*rowptr)[idx] = nnz;
            // if no upper boundary element, connect with upper neighbor
            if (i != 0)
            {
                (*col)[nnz] = idx - ndim;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if (j != 0)
            {
                (*col)[nnz] = idx - 1;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // element itself
            (*col)[nnz] = idx;
            (*val)[nnz] = 4.0;
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if (j != ndim - 1)
            {
                (*col)[nnz] = idx + 1;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if (i != ndim - 1)
            {
                (*col)[nnz] = idx + ndim;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
        }
    }
    (*rowptr)[n] = nnz;

    return n;
}

#endif // ROCSPARSE_BENCHMARK_UTILS_H_
