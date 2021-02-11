/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

template <typename I, typename J, typename T>
void testing_spmv_csr_bad_arg(const Arguments& arg)
{

    T alpha = 0.6;
    T beta  = 0.1;

    rocsparse_local_handle     local_handle;
    device_csr_matrix<T, I, J> dA;
    device_dense_matrix<T>     dx, dy;

    rocsparse_handle      handle  = local_handle;
    rocsparse_operation   trans   = rocsparse_operation_none;
    const void*           p_alpha = (const void*)&alpha;
    rocsparse_local_spmat A(dA);
    rocsparse_local_dnvec x(dx);
    const void*           p_beta = (const void*)&beta;
    rocsparse_local_dnvec y(dy);
    rocsparse_spmv_alg    alg = rocsparse_spmv_alg_default;
    size_t                buffer_size;
    size_t*               p_buffer_size = &buffer_size;
    void*                 temp_buffer   = (void*)0x4;
    rocsparse_datatype    ttype         = get_datatype<T>();

#define PARAMS                                                                                \
    handle, trans, p_alpha, (const rocsparse_spmat_descr&)A, (const rocsparse_dnvec_descr&)x, \
        p_beta, (rocsparse_dnvec_descr&)y, ttype, alg, p_buffer_size, temp_buffer

    static const int nex   = 2;
    static const int ex[2] = {9, 10};
    auto_testing_bad_arg(rocsparse_spmv, nex, ex, PARAMS);

    p_buffer_size = nullptr;
    temp_buffer   = nullptr;
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmv(PARAMS), rocsparse_status_invalid_pointer);

#undef PARAMS
}

template <typename I, typename J, typename T>
void testing_spmv_csr(const Arguments& arg)
{
    J                    M     = arg.M;
    J                    N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;
    rocsparse_spmv_alg   alg   = arg.spmv_alg;

    std::string filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    bool               adaptive = (alg == rocsparse_spmv_alg_csr_stream) ? false : true;
    rocsparse_datatype ttype    = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;
    using host_csr   = host_csr_matrix<T, I, J>;
    using device_csr = device_csr_matrix<T, I, J>;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

#define PARAMS(alpha_, A_, x_, beta_, y_) \
    handle, trans, alpha_, A_, x_, beta_, y_, ttype, alg, &buffer_size, dbuffer

    // Argument sanity check before allocating invalid memory

    // Allocate memory on device
    // Pointer mode

    // Check structures
    // Check SpMV when structures can be created

    if(M == 0 || N == 0)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        device_csr             dA;
        device_dense_matrix<T> dx, dy;

        rocsparse_local_spmat A(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

        size_t buffer_size;
        void*  dbuffer = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)),
                                rocsparse_status_success);
        CHECK_HIP_ERROR(hipMalloc(&dbuffer, 10));
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)),
                                rocsparse_status_success);
        CHECK_HIP_ERROR(hipFree(dbuffer));
        return;
    }

    // Wavefront size
    int dev;
    hipGetDevice(&dev);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    bool                              type      = (prop.warpSize == 32) ? true : adaptive;
    static constexpr bool             full_rank = false;
    rocsparse_matrix_factory<T, I, J> matrix_factory(arg, arg.timing ? false : type, full_rank);

    host_csr hA;
    matrix_factory.init_csr(hA, M, N);
    device_csr dA(hA);

    host_dense_matrix<T> hx(N, 1), hy(M, 1);

    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);
    device_dense_matrix<T> dx(hx), dy(hy);
    // Query SpMV buffer and allocate buffer

    rocsparse_local_spmat A(dA);
    rocsparse_local_dnvec x(dx);
    rocsparse_local_dnvec y(dy);

    void*  dbuffer = nullptr;
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)));
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)));

        // CPU csrmv
        {
            host_dense_matrix<T> hy_copy(hy);
            host_csrmv<I, J, T>(
                M, hA.nnz, *h_alpha, hA.ptr, hA.ind, hA.val, hx, *h_beta, hy, base, adaptive);
            hy.near_check(dy);
            dy.transfer_from(hy_copy);
        }

        // Pointer mode device
        {
            device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(PARAMS(d_alpha, A, x, d_beta, y)));
        }

        hy.near_check(dy);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(PARAMS(h_alpha, A, x, h_beta, y)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(dA.m, dA.nnz, *h_beta != static_cast<T>(0));
        double gbyte_count = csrmv_gbyte_count<T>(dA.m, dA.n, dA.nnz, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz",
                            dA.nnz,
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            "Algorithm",
                            (adaptive ? "adaptive" : "stream"),
                            "GFlop/s",
                            gpu_gflops,
                            "GB/s",
                            gpu_gbyte,
                            "msec",
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }

    CHECK_HIP_ERROR(hipFree(dbuffer));
    return;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                               \
    template void testing_spmv_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmv_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
