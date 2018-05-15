/* ************************************************************************
* Copyright 2015 Vratis, Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ************************************************************************ */

#pragma once
#ifndef CSRMV_DEVICE_H
#define CSRMV_DEVICE_H

#include <hip/hip_runtime.h>

// Knuth's Two-Sum algorithm, which allows us to add together two floating
// point numbers and exactly tranform the answer into a sum and a
// rounding error.
// Inputs: x and y, the two inputs to be aded together.
// In/Out: *sumk_err, which is incremented (by reference) -- holds the
//         error value as a result of the 2sum calculation.
// Returns: The non-corrected sum of inputs x and y.
template <typename T>
static __device__ T two_sum(T x, T y, T* sumk_err)
{
    const T sumk_s = x + y;
#ifdef EXTENDED_PRECISION
    // We use this 2Sum algorithm to perform a compensated summation,
    // which can reduce the cummulative rounding errors in our SpMV summation.
    // Our compensated sumation is based on the SumK algorithm (with K==2) from
    // Ogita, Rump, and Oishi, "Accurate Sum and Dot Product" in
    // SIAM J. on Scientific Computing 26(6) pp 1955-1988, Jun. 2005.

    // 2Sum can be done in 6 FLOPs without a branch. However, calculating
    // double precision is slower than single precision on every existing GPU.
    // As such, replacing 2Sum with Fast2Sum when using DPFP results in slightly
    // better performance. This is especially true on non-workstation GPUs with
    // low DPFP rates. Fast2Sum is faster even though we must ensure that
    // |a| > |b|. Branch divergence is better than the DPFP slowdown.
    // Thus, for DPFP, our compensated summation algorithm is actually described
    // by both Pichat and Neumaier in "Correction d'une somme en arithmetique
    // a virgule flottante" (J. Numerische Mathematik 19(5) pp. 400-406, 1972)
    // and "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
    // Summen (ZAMM Z. Angewandte Mathematik und Mechanik 54(1) pp. 39-51,
    // 1974), respectively.
    if(fabs(x) < fabs(y))
    {
        const T swap = x;

        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
// Original 6 FLOP 2Sum algorithm.
// T bp = sumk_s - x;
// (*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
#endif
    return sumk_s;
}

// Performs (x_vals * x_vec) + y using an FMA.
// Ideally, we would perform an error-free transformation here and return the
// appropriate error. However, the EFT of an FMA is very expensive. As such,
// if we are in EXTENDED_PRECISION mode, this function devolves into two_sum
// with x_vals and x_vec inputs multiplied separately from the compensated add.
template <typename T>
static __device__ T two_fma(T x_vals, T x_vec, T y, T* sumk_err)
{
#ifdef EXTENDED_PRECISION
    T x = x_vals * x_vec;

    const T sumk_s = x + y;
    if(fabs(x) < fabs(y))
    {
        const T swap = x;

        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
    // 2Sum in the FMA case. Poor performance on low-DPFP GPUs.
    // const T bp = fma(-x_vals, x_vec, sumk_s);
    // (*sumk_err) += (fma(x_vals, x_vec, -(sumk_s - bp)) + (y - bp));
    return sumk_s;
#else
    return fma(x_vals, x_vec, y);
#endif
}

// A method of doing the final reduction without having to copy and paste
// it a bunch of times.
// The EXTENDED_PRECISION section is done as part of the PSum2 addition,
// where we take temporary sums and errors for multiple threads and combine
// them together using the same 2Sum method.
// Inputs:  cur_sum: the input from which our sum starts
//          err: the current running cascade error for this final summation
//          partial: the local memory which holds the values to sum
//                  (we eventually use it to pass down temp. err vals as well)
//          lid: local ID of the work item calling this function.
//          thread_lane: The lane within this SUBWAVE for reduction.
//          round: This parallel summation method operates in multiple rounds
//                  to do a parallel reduction. See the blow comment for usage.
template <typename T, int SUBWAVE_SIZE>
static __device__ T
sum2_reduce(T cur_sum, T* err, volatile T* partial, int lid, int thread_lane, int round)
{
    if(SUBWAVE_SIZE > round)
    {
#ifdef EXTENDED_PRECISION
        const unsigned int partial_dest = lid + round;
        if(thread_lane < round)
            cur_sum = two_sum(cur_sum, partial[partial_dest], err);
        // We reuse the LDS entries to move the error values down into lower
        // threads. This saves LDS space, allowing higher occupancy, but requires
        // more barriers, which can reduce performance.
        __syncthreads();
        // Have all of those upper threads pass their temporary errors
        // into a location that the lower threads can read.
        if(thread_lane >= round)
            partial[lid] = *err;
        __syncthreads();
        if(thread_lane < round)
        { // Add those errors in.
            *err += partial[partial_dest];
            partial[lid] = cur_sum;
        }
#else
        // This is the more traditional reduction algorithm. It is up to
        // 25% faster (about 10% on average -- potentially worse on devices
        // with low double-precision calculation rates), but can result in
        // numerical inaccuracies, especially in single precision.
        cur_sum += partial[lid + round];
        __syncthreads();
        partial[lid] = cur_sum;
#endif
    }
    return cur_sum;
}

// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// int - typename for the type of integer data read by the kernel,  usually unsigned int
// T - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to
// process a single matrix row
template <typename T, int SUBWAVE_SIZE, int WG_SIZE>
static __device__ __launch_bounds__(WG_SIZE, 1) void csrmvn_general_device(int num_rows,
                                                                           T alpha,
                                                                           const int* row_offset,
                                                                           const int* col,
                                                                           const T* val,
                                                                           const T* x,
                                                                           T beta,
                                                                           T* y)
{
    __shared__ volatile T sdata[WG_SIZE + SUBWAVE_SIZE / 2];

    // const int vectors_per_block = WG_SIZE/SUBWAVE_SIZE;
    const int global_id   = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // global workitem id
    const int local_id    = hipThreadIdx_x;                                 // local workitem id
    const int thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const int vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    // const int vector_lane = local_id / SUBWAVE_SIZE;  // vector id within the workgroup
    const int num_vectors = hipGridDim_x * hipBlockDim_x / SUBWAVE_SIZE;

    for(int row = vector_id; row < num_rows; row += num_vectors)
    {
        const int row_start = row_offset[row];
        const int row_end   = row_offset[row + 1];

        T sum    = 0.;
        T sumk_e = 0.;

        // It is about 5% faster to always multiply by alpha, rather than to
        // check whether alpha is 0, 1, or other and do different code paths.
        for(int j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
            sum = two_fma(alpha * val[j], x[col[j]], sum, &sumk_e);

        T new_error = 0.;
        sum         = two_sum(sum, sumk_e, &new_error);

        // Parallel reduction in shared memory.
        sdata[local_id] = sum;

// This compensated summation reduces cummulative rounding errors,
// which can become a problem on GPUs because our reduction order is
// different than what would be used on a CPU.
// It is based on the PSumK algorithm (with K==2) from
// Yamanaka, Ogita, Rump, and Oishi, "A Parallel Algorithm of
// Accurate Dot Product," in the Journal of Parallel Computing,
// 34(6-8), pp. 392-410, Jul. 2008.
#pragma unroll
        for(int i = (WG_SIZE >> 1); i > 0; i >>= 1)
        {
            __syncthreads();
            sum = sum2_reduce<T, SUBWAVE_SIZE>(sum, &new_error, sdata, local_id, thread_lane, i);
        }

        if(thread_lane == 0)
        {
            if(beta == 0)
                y[row] = sum + new_error;
            else
            {
                sum    = two_fma(beta, y[row], sum, &new_error);
                y[row] = sum + new_error;
            }
        }
    }
}

#endif // CSRMV_DEVICE_H
