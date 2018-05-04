/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <stdlib.h>
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
void rocsparse_init_index(std::vector<I> &x, rocsparse_int nnz,
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
    std::sort(x.begin(), x.end());
};

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

        rocsparse_int N   = 128;
        rocsparse_int nnz = 32;

        double alpha = 1.0;

        rocsparse_index_base idxBase = rocsparse_index_base_zero;

        rocsparse_int norm_check = 0;
        rocsparse_int unit_check = 1;
        rocsparse_int timing     = 0;

        rocsparse_int iters = 10;

        Arguments& operator=(const Arguments& rhs)
        {
            N = rhs.N;
            nnz = rhs.nnz;

            alpha = rhs.alpha;

            idxBase = rhs.idxBase;

            norm_check = rhs.norm_check;
            unit_check = rhs.unit_check;
            timing     = rhs.timing;

            return *this;
        }
};

#endif // TESTING_UTILITY_HPP
