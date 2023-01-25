/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "auto_testing_bad_arg.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_matrix_type_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csr_matrix<U, I, J>;
};

template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csc, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csc_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csc_matrix<U, I, J>;
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_matrix<U, I>;
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_aos_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_aos_matrix<U, I>;
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_ell, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_ell_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_ell_matrix<U, I>;
};

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_sddmm_dispatch_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_csr, I, J, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csr, I, J, T>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<T, I, J>& matrix_factory,
                                      host_sparse_matrix<T>&             hA,
                                      Ts&&... ts)
    {
        matrix_factory.init_csr(hA, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<T>& hA,
                                 const host_dense_matrix<T>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<T>&      hC)
    {
        rocsparse_host<T, I, J>::csrddmm(trans_A,
                                         trans_B,
                                         hA.order,
                                         hB.order,
                                         hC.m,
                                         hC.n,
                                         ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
                                         hC.nnz,
                                         h_alpha,
                                         hA,
                                         hA.ld,
                                         hB,
                                         hB.ld,
                                         h_beta,
                                         hC.ptr,
                                         hC.ind,
                                         hC.val,
                                         hC.base);
    }
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_csc, I, J, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csc, I, J, T>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<T, I, J>& matrix_factory,
                                      host_sparse_matrix<T>&             hA,
                                      Ts&&... ts)
    {
        matrix_factory.init_csc(hA, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<T>& hA,
                                 const host_dense_matrix<T>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<T>&      hC)
    {
        rocsparse_host<T, I, J>::cscddmm(trans_A,
                                         trans_B,
                                         hA.order,
                                         hB.order,
                                         hC.m,
                                         hC.n,
                                         ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
                                         hC.nnz,
                                         h_alpha,
                                         hA,
                                         hA.ld,
                                         hB,
                                         hB.ld,
                                         h_beta,
                                         hC.ptr,
                                         hC.ind,
                                         hC.val,
                                         hC.base);
    }
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_coo, I, I, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo, I, I, T>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<T, I, I>& matrix_factory,
                                      host_sparse_matrix<T>&             hA,
                                      Ts&&... ts)
    {
        matrix_factory.init_coo(hA, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<T>& hA,
                                 const host_dense_matrix<T>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<T>&      hC)
    {
        rocsparse_host<T, I, I>::cooddmm(trans_A,
                                         trans_B,
                                         hA.order,
                                         hB.order,
                                         hC.m,
                                         hC.n,
                                         ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
                                         hC.nnz,
                                         h_alpha,
                                         hA,
                                         hA.ld,
                                         hB,
                                         hB.ld,
                                         h_beta,
                                         hC.row_ind,
                                         hC.col_ind,
                                         hC.val,
                                         hC.base);
    }
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_coo_aos, I, I, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, T>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<T, I, I>& matrix_factory,
                                      host_sparse_matrix<T>&             hA,
                                      Ts&&... ts)
    {
        matrix_factory.init_coo_aos(hA, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<T>& hA,
                                 const host_dense_matrix<T>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<T>&      hC)
    {
        rocsparse_host<T, I, I>::cooaosddmm(trans_A,
                                            trans_B,
                                            hA.order,
                                            hB.order,
                                            hC.m,
                                            hC.n,
                                            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
                                            hC.nnz,
                                            h_alpha,
                                            hA,
                                            hA.ld,
                                            hB,
                                            hB.ld,
                                            h_beta,
                                            hC.ind,
                                            hC.ind + 1,
                                            hC.val,
                                            hC.base);
    }
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_ell, I, I, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_ell, I, I, T>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<T, I, I>& matrix_factory,
                                      host_sparse_matrix<T>&             hA,
                                      Ts&&... ts)
    {
        matrix_factory.init_ell(hA, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<T>& hA,
                                 const host_dense_matrix<T>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<T>&      hC)
    {
        rocsparse_host<T, I, I>::ellddmm(trans_A,
                                         trans_B,
                                         hA.order,
                                         hB.order,
                                         hC.m,
                                         hC.n,
                                         ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
                                         hC.nnz,
                                         h_alpha,
                                         hA,
                                         hA.ld,
                                         hB,
                                         hB.ld,
                                         h_beta,
                                         hC.width,
                                         hC.ind,
                                         hC.val,
                                         hC.base);
    }
};

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_sddmm_dispatch
{
private:
    using traits = testing_sddmm_dispatch_traits<FORMAT, I, J, T>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

public:
    static void testing_sddmm_bad_arg(const Arguments& arg)
    {
        T alpha = 0.6;
        T beta  = 0.1;

        rocsparse_local_handle local_handle;

        rocsparse_handle    handle  = local_handle;
        rocsparse_operation trans_A = rocsparse_operation_none;
        rocsparse_operation trans_B = rocsparse_operation_none;
        const void*         p_alpha = (const void*)&alpha;
        const void*         p_beta  = (const void*)&beta;
        rocsparse_sddmm_alg alg     = rocsparse_sddmm_alg_default;
        size_t              buffer_size;
        size_t*             p_buffer_size = &buffer_size;
        void*               temp_buffer   = (void*)0x4;
        rocsparse_datatype  ttype         = get_datatype<T>();

#define PARAMS_BUFFER_SIZE                                                              \
    handle, trans_A, trans_B, p_alpha, (const rocsparse_dnmat_descr&)A,                 \
        (const rocsparse_dnmat_descr&)B, p_beta, (rocsparse_spmat_descr&)C, ttype, alg, \
        p_buffer_size

#define PARAMS                                                                          \
    handle, trans_A, trans_B, p_alpha, (const rocsparse_dnmat_descr&)A,                 \
        (const rocsparse_dnmat_descr&)B, p_beta, (rocsparse_spmat_descr&)C, ttype, alg, \
        temp_buffer

        //
        // AUTOMATIC BAD ARGS.
        //
        {
            device_dense_matrix<T> dA, dB;
            rocsparse_local_dnmat  A(dA), B(dB);

            device_sparse_matrix<T> dC;
            rocsparse_local_spmat   C(dC);

            static constexpr int num_to_exclude             = 1;
            static constexpr int to_exclude[num_to_exclude] = {10};
            auto_testing_bad_arg(rocsparse_sddmm_buffer_size, PARAMS_BUFFER_SIZE);
            auto_testing_bad_arg(rocsparse_sddmm_preprocess, num_to_exclude, to_exclude, PARAMS);
            auto_testing_bad_arg(rocsparse_sddmm, num_to_exclude, to_exclude, PARAMS);
        }

#undef PARAMS
#undef PARAMS_BUFFER_SIZE
    }

    static void testing_sddmm(const Arguments& arg)
    {
        J                    M       = arg.M;
        J                    N       = arg.N;
        J                    K       = arg.K;
        rocsparse_operation  trans_A = arg.transA;
        rocsparse_operation  trans_B = arg.transB;
        rocsparse_index_base base    = arg.baseA;
        rocsparse_sddmm_alg  alg     = rocsparse_sddmm_alg_default;
        rocsparse_datatype   ttype   = get_datatype<T>();
        rocsparse_order      order_A = arg.order;
        rocsparse_order      order_B = arg.order;

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());

#define PARAMS_BUFFER_SIZE(alpha_, A_, B_, beta_, C_)                                          \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        &buffer_size
#define PARAMS(alpha_, A_, B_, beta_, C_)                                                      \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        dbuffer

        // Argument sanity check before allocating invalid memory

        // Allocate memory on device
        // Pointer mode

        // Check structures
        // Check Sddmm when structures can be created
        if(M <= 0 || N <= 0)
        {

            if(M == 0 || N == 0)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                device_sparse_matrix<T> dC;
                device_dense_matrix<T>  dA, dB;

                rocsparse_local_spmat C(dC);
                rocsparse_local_dnmat A(dA), B(dB);

                size_t buffer_size;
                void*  dbuffer = nullptr;
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(h_alpha, A, B, h_beta, C)),
                    rocsparse_status_success);
                CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, 10));
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_sddmm_preprocess(PARAMS(h_alpha, A, B, h_beta, C)),
                    rocsparse_status_success);
                EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS(h_alpha, A, B, h_beta, C)),
                                        rocsparse_status_success);
                CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
                return;
            }
            return;
        }

        // Wavefront size
        int dev;
        hipGetDevice(&dev);

        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<T> hC;
        {
            bool                              type      = (prop.warpSize == 32) ? true : false;
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<T, I, J> matrix_factory(
                arg, arg.timing ? false : type, full_rank);
            traits::sparse_initialization(matrix_factory, hC, M, N, base);
        }

        device_sparse_matrix<T> dC(hC);

        const J              hA_m = (trans_A == rocsparse_operation_none) ? M : K;
        const J              hA_n = (trans_A == rocsparse_operation_none) ? K : M;
        host_dense_matrix<T> hA(hA_m, hA_n, order_A);
        rocsparse_matrix_utils::init_exact(hA);

        const J              hB_m = (trans_B == rocsparse_operation_none) ? K : N;
        const J              hB_n = (trans_B == rocsparse_operation_none) ? N : K;
        host_dense_matrix<T> hB(hB_m, hB_n, order_B);
        rocsparse_matrix_utils::init_exact(hB);

        device_dense_matrix<T> dA(hA), dB(hB);

        rocsparse_local_spmat C(dC);
        rocsparse_local_dnmat A(dA), B(dB);

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(h_alpha, A, B, h_beta, C)));
        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, std::max(buffer_size, sizeof(I))));
        if(arg.unit_check)
        {
            // Pointer mode host
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_sddmm_preprocess(PARAMS(h_alpha, A, B, h_beta, C)));

            CHECK_ROCSPARSE_ERROR(testing::rocsparse_sddmm(PARAMS(h_alpha, A, B, h_beta, C)));

            {
                host_vector<T> hC_val_copy(hC.val);
                //
                // HOST CALCULATION
                //
                traits::host_calculation(trans_A, trans_B, h_alpha, hA, hB, h_beta, hC);

                //
                // CHECK MATRICES
                //
                hC.near_check(dC);

                //
                // COPY BACK ORIGINAL VALUES.
                //
                dC.val.transfer_from(hC_val_copy);
            }

            // Pointer mode device
            {
                device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                CHECK_ROCSPARSE_ERROR(rocsparse_sddmm_preprocess(PARAMS(d_alpha, A, B, d_beta, C)));
                CHECK_ROCSPARSE_ERROR(testing::rocsparse_sddmm(PARAMS(d_alpha, A, B, d_beta, C)));
            }

            hC.near_check(dC);
        }

        if(arg.timing)
        {
            int number_cold_calls = 2;
            int number_hot_calls  = arg.iters;

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            // Warm up
            for(int iter = 0; iter < number_cold_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_sddmm_preprocess(PARAMS(h_alpha, A, B, h_beta, C)));
                CHECK_ROCSPARSE_ERROR(rocsparse_sddmm(PARAMS(h_alpha, A, B, h_beta, C)));
            }

            double gpu_time_used = get_time_us();

            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_sddmm(PARAMS(h_alpha, A, B, h_beta, C)));
            }

            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

            double gflop_count = rocsparse_gflop_count<FORMAT>::sddmm(
                dC.m, dC.n, dC.nnz, K, *h_beta != static_cast<T>(0));
            double gbyte_count = rocsparse_gbyte_count<FORMAT>::template sddmm<T>(
                dC.m, dC.n, dC.nnz, K, *h_beta != static_cast<T>(0));

            double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
            double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

            display_timing_info("format",
                                rocsparse_format2string(FORMAT),
                                "transA",
                                rocsparse_operation2string(trans_A),
                                "transB",
                                rocsparse_operation2string(trans_B),
                                "M",
                                M,
                                "N",
                                N,
                                "K",
                                K,
                                "nnz",
                                dC.nnz,
                                "alpha",
                                *h_alpha,
                                "beta",
                                *h_beta,
                                s_timing_info_perf,
                                gpu_gflops,
                                s_timing_info_bandwidth,
                                gpu_gbyte,
                                s_timing_info_time,
                                get_gpu_time_msec(gpu_time_used));
        }

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        return;
    }
};
