/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <rocsparse.h>
#include <hip/hip_runtime_api.h>

/*!\file
 * \brief provide data initialization and timing utilities.
 */

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#define CHECK_ROCSPARSE_ERROR(error)                              \
    if(error != rocsparse_status_success)                         \
    {                                                             \
        fprintf(stderr, "rocSPARSE error: ");                     \
        if(error == rocsparse_status_invalid_handle)              \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_handle");   \
        }                                                         \
        else if(error == rocsparse_status_not_implemented)        \
        {                                                         \
            fprintf(stderr, " rocsparse_status_not_implemented"); \
        }                                                         \
        else if(error == rocsparse_status_invalid_pointer)        \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_pointer");  \
        }                                                         \
        else if(error == rocsparse_status_invalid_size)           \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_size");     \
        }                                                         \
        else if(error == rocsparse_status_memory_error)           \
        {                                                         \
            fprintf(stderr, "rocsparse_status_memory_error");     \
        }                                                         \
        else if(error == rocsparse_status_internal_error)         \
        {                                                         \
            fprintf(stderr, "rocsparse_status_internal_error");   \
        }                                                         \
        else                                                      \
        {                                                         \
            fprintf(stderr, "rocsparse_status error");            \
        }                                                         \
        fprintf(stderr, "\n");                                    \
        return error;                                             \
    }

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number between [0, 0.999...] . */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return (T)(rand() % 10 + 1); // generate a integer number between [1, 10]
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void rocsparse_init(std::vector<T>& A, rocsparse_int M, rocsparse_int N)
{
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            A[i + j] = random_generator<T>();
        }
    }
};

/* ============================================================================================ */
/*! \brief  vector initialization: */
// initialize sparse index vector with nnz entries ranging from start to end
template <typename I>
void rocsparse_init_index(I *x, rocsparse_int nnz,
                          rocsparse_int start, rocsparse_int end)
{
    std::vector<bool> check(end-start, false);
    int num = 0;
    while (num < nnz)
    {
        rocsparse_int val = start + rand() % (end-start);
        if (!check[val-start])
        {
            x[num] = val;
            check[val-start] = true;
            ++num;
        }
    }
    std::sort(x, x+nnz);
};

/* ============================================================================================ */
/*! \brief  csr matrix initialization */
template <typename T>
void rocsparse_init_csr(std::vector<rocsparse_int> &ptr, std::vector<rocsparse_int> &col,
                        std::vector<T> &val,
                        rocsparse_int nrow, rocsparse_int ncol, rocsparse_int nnz)
{
    // Row offsets
    ptr[0] = 0;
    ptr[nrow] = nnz;

    for (rocsparse_int i=1; i<nrow; ++i)
    {
        ptr[i] = rand() % (nnz-1) + 1;
    }
    std::sort(ptr.begin(), ptr.end());

    // Column indices
    for (rocsparse_int i=0; i<nrow; ++i)
    {
        rocsparse_init_index(&col[ptr[i]], ptr[i+1]-ptr[i], 0, ncol-1);
        std::sort(&col[ptr[i]], &col[ptr[i+1]]);
    }

    // Random values
    for (rocsparse_int i=0; i<nnz; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename T>
rocsparse_int gen_2d_laplacian(rocsparse_int ndim,
                               std::vector<rocsparse_int> &rowptr,
                               std::vector<rocsparse_int> &col,
                               std::vector<T> &val)
{
    if (ndim == 0) {
        return 0;
    }

    rocsparse_int n = ndim * ndim;
    rocsparse_int nnz_mat = n * 5 - ndim * 4;

    rowptr.resize(n+1);
    col.resize(nnz_mat);
    val.resize(nnz_mat);

    rocsparse_int nnz = 0;

    // Fill local arrays
    for (rocsparse_int i=0; i<ndim; ++i)
    {
        for (rocsparse_int j=0; j<ndim; ++j)
        {
            rocsparse_int idx = i*ndim+j;
            rowptr[idx] = nnz;
            // if no upper boundary element, connect with upper neighbor
            if (i != 0)
            {
                col[nnz] = idx - ndim;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if (j != 0)
            {
                col[nnz] = idx - 1;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // element itself
            col[nnz] = idx;
            val[nnz] = static_cast<T>(4);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if (j != ndim - 1)
            {
                col[nnz] = idx + 1;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if (i != ndim - 1)
            {
                col[nnz] = idx + ndim;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
rocsparse_int gen_matrix_coo(rocsparse_int m,
                             rocsparse_int n,
                             rocsparse_int nnz,
                             std::vector<rocsparse_int> &row_ind,
                             std::vector<rocsparse_int> &col_ind,
                             std::vector<T> &val)
{
    if (row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if (col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if (val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Uniform distributed row indices
    for (rocsparse_int i=0; i<nnz; ++i)
    {
        row_ind[i] = rand() % m;
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    rocsparse_int i=0;
    while (i < nnz)
    {
        rocsparse_int begin = i;
        while (row_ind[i] == row_ind[begin])
            ++i;

        // Sample i disjunct column indices
        rocsparse_int idx = begin;
        while (idx < i)
        {
            // Normal distribution around the diagonal
            rocsparse_int rng = row_ind[begin] + (i - begin)
                              * sqrt(-2.0 * log((double) rand() / RAND_MAX))
                              * cos(2.0 * M_PI * (double) rand() / RAND_MAX);

            // Repeat if running out of bounds
            if (rng < 0 || rng > n-1)
                continue;

            // Check for disjunct column index in current row
            if (!check[rng])
            {
                check[rng] = true;
                col_ind[idx] = rng;
                ++idx;
            }
        }

        // Reset disjunct check array
        for (rocsparse_int j=begin; j<i; ++j)
            check[col_ind[j]] = false;

        // Partially sort column indices
        std::sort(&col_ind[begin], &col_ind[i]);
    }

    // Sample random values
    for (rocsparse_int i=0; i<nnz; ++i)
    {
        val[i] = (double) rand() / RAND_MAX;
    }

}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename T>
rocsparse_int read_mtx_matrix(const char *filename,
                              rocsparse_int &nrow,
                              rocsparse_int &ncol,
                              rocsparse_int &nnz,
                              std::vector<rocsparse_int> &row,
                              std::vector<rocsparse_int> &col,
                              std::vector<T> &val)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        return -1;
    }

    char line[1024];

    // Check for banner
    if (!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if (sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for (char *p=array; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=coord; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=data; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=type; *p!='\0'; *p=tolower(*p), p++);

    // Check banner
    if (strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if (strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if (strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if (strcmp(data, "real") != 0)
    {
        return -1;
    }

    // Check type
    if (strcmp(type, "general") != 0 &&
        strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    rocsparse_int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if (line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    rocsparse_int snnz;

    sscanf(line, "%d %d %d", &nrow, &ncol, &snnz);
    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Read entries
    rocsparse_int idx = 0;
    while(fgets(line, 1024, f))
    {
        rocsparse_int irow;
        rocsparse_int icol;
        double dval;

        sscanf(line, "%d %d %lf", &irow, &icol, &dval);

        --irow;
        --icol;

        row[idx] = irow;
        col[idx] = icol;
        val[idx] = (T) dval;

        ++idx;

        if (symm && irow != icol) {

            row[idx] = icol;
            col[idx] = irow;
            val[idx] = (T) dval;

            ++idx;

        }

    }
    fclose(f);

    return 0;
}

/* ============================================================================================ */
/*! \brief  Convert matrix from COO to CSR format */
template <typename T>
void coo_to_csr(rocsparse_int nrow, rocsparse_int ncol, rocsparse_int nnz,
                const std::vector<rocsparse_int> &src_row,
                const std::vector<rocsparse_int> &src_col,
                const std::vector<T> &src_val,
                std::vector<rocsparse_int> &dst_ptr,
                std::vector<rocsparse_int> &dst_col,
                std::vector<T> &dst_val)
{
    dst_ptr.resize(nrow+1, 0);
    dst_col.resize(nnz);
    dst_val.resize(nnz);

    // Compute nnz entries per row
    for (rocsparse_int i=0; i<nnz; ++i)
    {
        ++dst_ptr[src_row[i]];
    }

    rocsparse_int sum = 0;
    for (rocsparse_int i=0; i<nrow; ++i)
    {
        rocsparse_int tmp = dst_ptr[i];
        dst_ptr[i] = sum;
        sum += tmp;
    }
    dst_ptr[nrow] = sum;

    // Write column index and values
    for (rocsparse_int i=0; i<nnz; ++i)
    {
        rocsparse_int row = src_row[i];
        rocsparse_int idx = dst_ptr[row];

        dst_col[idx] = src_col[i];
        dst_val[idx] = src_val[i];

        ++dst_ptr[row];
    }

    rocsparse_int last = 0;
    for (rocsparse_int i=0; i<nrow+1; ++i)
    {
        rocsparse_int tmp = dst_ptr[i];
        dst_ptr[i] = last;
        last = tmp;
    }

    for (rocsparse_int i=0; i<nrow; ++i)
    {
        for (rocsparse_int j=dst_ptr[i]; j<dst_ptr[i+1]; ++j)
        {
            for (rocsparse_int k=dst_ptr[i]; k<dst_ptr[i+1]-1; ++k)
            {
                // Swap elements
                rocsparse_int idx = dst_col[k];
                T val = dst_val[k];

                dst_col[k] = dst_col[k+1];
                dst_val[k] = dst_val[k+1];

                dst_col[k+1] = idx;
                dst_val[k+1] = val;
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  device query and print out their ID and name */
rocsparse_int query_device_property();

/*  set current device to device_id */
void set_device(rocsparse_int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this rocsparse library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
    public:

        rocsparse_int M   = 128;
        rocsparse_int N   = 128;
        rocsparse_int nnz = 32;

        double alpha = 1.0;
        double beta  = 0.0;

        rocsparse_operation trans = rocsparse_operation_none;
        rocsparse_index_base idxBase = rocsparse_index_base_zero;

        rocsparse_int norm_check = 0;
        rocsparse_int unit_check = 1;
        rocsparse_int timing     = 0;

        rocsparse_int iters = 10;

        std::string filename = "";

        Arguments& operator=(const Arguments& rhs)
        {
            M = rhs.M;
            N = rhs.N;
            nnz = rhs.nnz;

            alpha = rhs.alpha;
            beta  = rhs.beta;

            trans = rhs.trans;
            idxBase = rhs.idxBase;

            norm_check = rhs.norm_check;
            unit_check = rhs.unit_check;
            timing     = rhs.timing;

            filename = rhs.filename;

            return *this;
        }
};

#endif // TESTING_UTILITY_HPP
