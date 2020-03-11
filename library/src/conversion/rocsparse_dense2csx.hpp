/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "dense2csx_device.h"

template <rocsparse_direction DIRA, typename T>
rocsparse_status rocsparse_dense2csx_template(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              const rocsparse_mat_descr descrA,
                                              const T*                  A,
                                              rocsparse_int             lda,
                                              T*                        csxValA,
                                              rocsparse_int*            csxRowColPtrA,
                                              rocsparse_int*            csxColRowIndA)
{
    if(0 == m || 0 == n)
    {
        return rocsparse_status_success;
    }

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);
    hipStream_t                    stream     = handle->stream;
    switch(DIRA)
    {
    case rocsparse_direction_row:
    {

        static constexpr rocsparse_int WF_SIZE         = 64;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        rocsparse_int                  blocks          = (m - 1) / NROWS_PER_BLOCK + 1;
        dim3                           k_blocks(blocks), k_threads(WF_SIZE * NROWS_PER_BLOCK);
        hipLaunchKernelGGL((dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
                           k_blocks,
                           k_threads,
                           0,
                           stream,
                           descrA->base,
                           m,
                           n,
                           A,
                           lda,
                           csxValA,
                           csxRowColPtrA,
                           csxColRowIndA);
        return rocsparse_status_success;
    }

    case rocsparse_direction_column:
    {
        static constexpr rocsparse_int WF_SIZE            = 64;
        static constexpr rocsparse_int NCOLUMNS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        rocsparse_int                  blocks             = (n - 1) / NCOLUMNS_PER_BLOCK + 1;
        dim3                           k_blocks(blocks), k_threads(WF_SIZE * NCOLUMNS_PER_BLOCK);
        hipLaunchKernelGGL((dense2csc_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE, T>),
                           k_blocks,
                           k_threads,
                           0,
                           stream,
                           descrA->base,
                           m,
                           n,
                           A,
                           lda,
                           csxValA,
                           csxRowColPtrA,
                           csxColRowIndA);

        return rocsparse_status_success;
    }
    }

    return rocsparse_status_invalid_value;
}
