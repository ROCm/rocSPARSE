/* ************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename I, typename T>
void testing_spmm_bell_bad_arg(const Arguments& arg)
{
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle      handle      = local_handle;
    rocsparse_operation   trans_A     = rocsparse_operation_none;
    rocsparse_operation   trans_B     = rocsparse_operation_none;
    void*                 alpha       = (void*)0x4;
    rocsparse_spmat_descr A           = (rocsparse_spmat_descr)0x4;
    rocsparse_dnmat_descr B           = (rocsparse_dnmat_descr)0x4;
    void*                 beta        = (void*)0x4;
    rocsparse_dnmat_descr C           = (rocsparse_dnmat_descr)0x4;
    rocsparse_datatype    ttype       = rocsparse_datatype_f32_r;
    rocsparse_spmm_alg    alg         = rocsparse_spmm_alg_bell;
    rocsparse_spmm_stage  stage       = rocsparse_spmm_stage_auto;
    size_t*               buffer_size = (size_t*)0x4;
    void*                 buffer      = (void*)0x4;

#define PARAMS \
    handle, trans_A, trans_B, &alpha, A, B, &beta, C, ttype, alg, stage, buffer_size, buffer

    static const int nargs_to_exclude                  = 2;
    static const int args_to_exclude[nargs_to_exclude] = {11, 12};

    auto_testing_bad_arg(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);

#undef PARAMS
}

template <typename I, typename T>
void testing_spmm_bell(const Arguments& arg)
{
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    I M         = arg.M;
    I N         = arg.N;
    I K         = arg.K;
    I block_dim = arg.block_dim;

    rocsparse_operation  trans_A   = arg.transA;
    rocsparse_operation  trans_B   = arg.transB;
    rocsparse_direction  direction = arg.direction;
    rocsparse_index_base base      = arg.baseA;
    rocsparse_spmm_alg   alg       = arg.spmm_alg;
    rocsparse_order      order     = arg.order;

    I Mb = -1, Nb = -1, Kb = -1;

    host_scalar<T> h_alpha, h_beta;

    *h_alpha = arg.get_alpha<T>();
    *h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K <= 0 || block_dim <= 0)
    {
        const rocsparse_int safe_width = 4;
        // Pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        I nrow_A = trans_A == rocsparse_operation_none ? M : K;
        I ncol_A = trans_A == rocsparse_operation_none ? K : M;

        rocsparse_local_spmat A(nrow_A,
                                ncol_A,
                                direction,
                                block_dim,
                                std::min(safe_width * block_dim, ncol_A),
                                (I*)0x4,
                                (T*)0x4,
                                itype,
                                base,
                                ttype);
        I                     ldb = order == rocsparse_order_column
                                        ? (trans_B == rocsparse_operation_none ? 2 * K : 2 * N)
                                        : (trans_B == rocsparse_operation_none ? 2 * N : 2 * K);

        I nrow_B = trans_B == rocsparse_operation_none ? K : N;
        I ncol_B = trans_B == rocsparse_operation_none ? N : K;

        I                     ldc    = order == rocsparse_order_column ? 2 * M : 2 * N;
        I                     nrow_C = M;
        I                     ncol_C = N;
        rocsparse_local_dnmat B(nrow_B, ncol_B, ldb, (void*)0x4, ttype, order);
        rocsparse_local_dnmat C(nrow_C, ncol_C, ldc, (void*)0x4, ttype, order);

        void*  dbuffer     = nullptr;
        size_t buffer_size = sizeof(I);
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha,
                                               A,
                                               B,
                                               h_beta,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spmm_stage_buffer_size,
                                               &buffer_size,
                                               dbuffer),
                                (M < 0 || N < 0 || K < 0 || block_dim <= 0)
                                    ? rocsparse_status_invalid_pointer
                                    : rocsparse_status_success);

        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha,
                                               A,
                                               B,
                                               h_beta,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spmm_stage_preprocess,
                                               &buffer_size,
                                               dbuffer),
                                (M < 0 || N < 0 || K < 0 || block_dim <= 0)
                                    ? rocsparse_status_invalid_pointer
                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha,
                                               A,
                                               B,
                                               h_beta,
                                               C,
                                               ttype,
                                               alg,
                                               rocsparse_spmm_stage_compute,
                                               &buffer_size,
                                               dbuffer),
                                (M < 0 || N < 0 || K < 0 || block_dim <= 0)
                                    ? rocsparse_status_invalid_pointer
                                    : rocsparse_status_success);
        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        return;
    }

    rocsparse_matrix_factory<T, I, I> matrix_factory(arg);

    host_ell_matrix<T, I> hA;
    I                     hA_m = (trans_A == rocsparse_operation_none) ? M : K;
    I                     hA_n = (trans_A == rocsparse_operation_none) ? K : M;
    matrix_factory.init_ell(hA, hA_m, hA_n, base);

    Mb = hA_m;
    Nb = N;
    Kb = hA_n;

    M = Mb * block_dim;
    N = Nb * block_dim;
    K = Kb * block_dim;

    host_dense_matrix<T> hB((trans_B == rocsparse_operation_none) ? K : N,
                            (trans_B == rocsparse_operation_none) ? N : K);
    rocsparse_matrix_utils::init_exact(hB);
    device_dense_matrix<T> dB(hB);

    //
    // C
    //
    host_dense_matrix<T> hC(M, N);
    rocsparse_matrix_utils::init_exact(hC);
    device_dense_matrix<T> dC(hC);

    device_ell_matrix<T, I> dA(hA);
    host_dense_matrix<T>    hA_val(1, dA.width * Mb * block_dim * block_dim);
    rocsparse_matrix_utils::init_exact(hA_val);
    device_dense_matrix<T> dA_val(hA_val);
    rocsparse_local_spmat  A(M,
                            K,
                            direction,
                            block_dim,
                            dA.width * block_dim,
                            (I*)dA.ind,
                            (T*)dA_val,
                            itype,
                            base,
                            ttype);

    rocsparse_local_dnmat B(
        dB.m, dB.n, (order == rocsparse_order_column) ? dB.m : dB.n, dB, ttype, order);
    rocsparse_local_dnmat C(
        dC.m, dC.n, (order == rocsparse_order_column) ? dC.m : dC.n, dC, ttype, order);

    // Query SpMM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         h_alpha,
                                         A,
                                         B,
                                         h_beta,
                                         C,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         h_alpha,
                                         A,
                                         B,
                                         h_beta,
                                         C,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    if(arg.unit_check)
    {
        // SpMM

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             trans_A,
                                             trans_B,
                                             h_alpha,
                                             A,
                                             B,
                                             h_beta,
                                             C,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             dbuffer));
        {
            host_dense_matrix<T> hC_copy(hC);

            //
            // Host calculation.
            //
            // Convert the Blocked ELL matrix to coo matrix...
            //
            I nnzb  = 0;
            I bound = hA.m * hA.width;
            for(size_t i = 0; i < bound; ++i)
            {
                if(hA.ind[i] - hA.base >= 0)
                {
                    ++nnzb;
                }
            }

            I* coo_row = new I[size_t(nnzb) * block_dim * block_dim];
            I* coo_col = new I[size_t(nnzb) * block_dim * block_dim];
            T* coo_val = new T[size_t(nnzb) * block_dim * block_dim];

            size_t at = 0;
            for(I ib = 0; ib < Mb; ++ib)
            {
                for(I l = 0; l < hA.width; ++l)
                {
                    const size_t idx = Mb * l + ib;

                    const I jb = hA.ind[idx] - hA.base;
                    if(jb >= 0)
                    {
                        if(direction == rocsparse_direction_column)
                        {
                            for(I lcol = 0; lcol < block_dim; ++lcol)
                            {
                                for(I lrow = 0; lrow < block_dim; ++lrow)
                                {
                                    coo_row[at] = ib * block_dim + lrow + hA.base;
                                    coo_col[at] = jb * block_dim + lcol + hA.base;
                                    coo_val[at] = hA_val[block_dim * block_dim * idx
                                                         + lcol * block_dim + lrow];
                                    ++at;
                                }
                            }
                        }
                        else
                        {
                            for(I lcol = 0; lcol < block_dim; ++lcol)
                            {
                                for(I lrow = 0; lrow < block_dim; ++lrow)
                                {
                                    coo_row[at] = ib * block_dim + lrow + hA.base;
                                    coo_col[at] = jb * block_dim + lcol + hA.base;
                                    coo_val[at] = hA_val[block_dim * block_dim * idx
                                                         + lrow * block_dim + lcol];
                                    ++at;
                                }
                            }
                        }
                    }
                }
            }

            host_coomm<T, I>(M,
                             N,
                             K,
                             nnzb * block_dim * block_dim,
                             trans_A,
                             trans_B,
                             *h_alpha,
                             coo_row,
                             coo_col,
                             coo_val,
                             hB,
                             (order == rocsparse_order_column) ? hB.m : hB.n,
                             *h_beta,
                             hC,
                             (order == rocsparse_order_column) ? hC.m : hC.n,
                             order,
                             base);

            delete[] coo_val;
            delete[] coo_col;
            delete[] coo_row;
            if(trans_A == rocsparse_operation_none)
            {
                hC.near_check(dC);
            }
            dC = hC_copy;
        }

        // Pointer mode device
        {

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
            CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 d_alpha,
                                                 A,
                                                 B,
                                                 d_beta,
                                                 C,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 h_alpha,
                                                 A,
                                                 B,
                                                 h_beta,
                                                 C,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 h_alpha,
                                                 A,
                                                 B,
                                                 h_beta,
                                                 C,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;
        double gflop_count
            = spmm_gflop_count(N, dA.nnz, (I)dC.m * (I)dC.n, *h_beta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrmm_gbyte_count<T>(
            dA.m, dA.nnz, (I)dB.m * (I)dB.n, (I)dC.m * (I)dC.n, *h_beta != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "K",
                            K,
                            "nnz",
                            dA.nnz,
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            "Algorithm",
                            rocsparse_spmmalg2string(alg),
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            gpu_time_used / 1e3);
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                \
    template void testing_spmm_bell_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmm_bell<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
