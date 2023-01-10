/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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

#include "testing_csritilu0.hpp"

#include "rocsparse.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"
#include <hip/hip_runtime.h>
template <typename T, typename I, typename J>
rocsparse_status rocsparse_host_csrscaling_ruiz(
    J m_, J n_, I nnz_, const I* ptr_, const J* ind_, T* val_, rocsparse_index_base base_)

{
    T*      D1   = (T*)malloc(sizeof(T) * m_);
    T*      D2   = (T*)malloc(sizeof(T) * n_);
    T*      DR   = (T*)malloc(sizeof(T) * m_);
    T*      DC   = (T*)malloc(sizeof(T) * n_);
    double* DRMX = (double*)malloc(sizeof(double) * m_);
    double* DCMX = (double*)malloc(sizeof(double) * n_);
    for(J i = 0; i < m_; ++i)
        D1[i] = static_cast<T>(1);
    for(J j = 0; j < n_; ++j)
        D2[j] = static_cast<T>(1);
    for(J i = 0; i < m_; ++i)
        DRMX[i] = static_cast<double>(0);
    for(J j = 0; j < n_; ++j)
        DCMX[j] = static_cast<double>(0);
    for(J iter = 0; iter < 40; ++iter)
    {
        for(J i = 0; i < m_; ++i)
            DRMX[i] = static_cast<double>(0);
        for(J j = 0; j < n_; ++j)
            DCMX[j] = static_cast<double>(0);

        for(J i = 0; i < m_; ++i)
        {
            for(J k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
            {
                const J      j = ind_[k] - base_;
                const double v = std::abs(val_[k]);

                DRMX[i] = std::max(DRMX[i], v);
                DCMX[j] = std::max(DCMX[j], v);
            }
        }

        double resr = 0.0;
        double resc = 0.0;
        for(J i = 0; i < m_; ++i)
            resr = std::max(resr, std::abs((double(1.0) - DRMX[i])));
        for(J j = 0; j < n_; ++j)
            resc = std::max(resc, std::abs((double(1.0) - DCMX[j])));
        if(resr < 1e-5 && resc < 1.0e-5)
            break;

        for(J i = 0; i < m_; ++i)
            DRMX[i] = static_cast<double>(1) / sqrt(DRMX[i]);
        for(J j = 0; j < n_; ++j)
            DCMX[j] = static_cast<double>(1) / sqrt(DCMX[j]);
        for(J i = 0; i < m_; ++i)
        {
            for(J k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
            {
                const J j = ind_[k] - base_;
                val_[k] *= DRMX[i] * DCMX[j];
            }
        }

        for(J i = 0; i < m_; ++i)
            D1[i] *= DRMX[i];
        for(J j = 0; j < n_; ++j)
            D2[j] *= DCMX[j];
    }
    free(DRMX);
    free(DC);
    free(DR);
    free(D1);
    free(D2);
    return rocsparse_status_success;
}

template <typename T>
static rocsparse_status csrilu0(rocsparse_handle          handle_,
                                device_csr_matrix<T>&     dA_,
                                rocsparse_mat_descr       descr_,
                                rocsparse_analysis_policy apol_,
                                rocsparse_solve_policy    spol_,
                                rocsparse_mat_info        info_,
                                int*                      pivot_,
                                rocsparse_status&         status_csrilu0_buffer_size,
                                rocsparse_status&         status_csrilu0_analysis,
                                rocsparse_status&         status_csrilu0)
{
    rocsparse_status status;

    // Obtain required buffer size
    size_t buffer_size;
    status = rocsparse_csrilu0_buffer_size<T>(
        handle_, dA_.m, dA_.nnz, descr_, dA_.val, dA_.ptr, dA_.ind, info_, &buffer_size);
    status_csrilu0_buffer_size = status;
    if(status != rocsparse_status_success)
    {
        return status;
    }

    void* dbuffer;
    rocsparse_hipMalloc(&dbuffer, buffer_size);
    status = rocsparse_set_pointer_mode(handle_, rocsparse_pointer_mode_host);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_csrilu0_analysis<T>(
        handle_, dA_.m, dA_.nnz, descr_, dA_.val, dA_.ptr, dA_.ind, info_, apol_, spol_, dbuffer);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status                  = rocsparse_csrilu0_zero_pivot(handle_, info_, pivot_);
    status_csrilu0_analysis = status;
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status = rocsparse_csrilu0<T>(
        handle_, dA_.m, dA_.nnz, descr_, dA_.val, dA_.ptr, dA_.ind, info_, spol_, dbuffer);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    status         = rocsparse_csrilu0_zero_pivot(handle_, info_, pivot_);
    status_csrilu0 = status;
    if(status != rocsparse_status_success)
    {
        return status;
    }

    rocsparse_csrilu0_clear(handle_, info_);

    if(hipSuccess != hipFree(dbuffer))
    {
        return rocsparse_status_memory_error;
    }
    return rocsparse_status_success;
}

void testing_csritilu0_extra(const Arguments& arg) {}

template <typename T>
void testing_csritilu0_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle   = local_handle;
    rocsparse_itilu0_alg   alg      = rocsparse_itilu0_alg_default;
    rocsparse_int          option   = 0;
    rocsparse_int          maxiter  = 1;
    rocsparse_int          m        = 1;
    rocsparse_int          nnz      = 1;
    rocsparse_int*         ptr      = (rocsparse_int*)0x4;
    rocsparse_int*         ind      = (rocsparse_int*)0x4;
    rocsparse_index_base   base     = rocsparse_index_base_zero;
    rocsparse_datatype     datatype = rocsparse_datatype_t::get<T>();
    size_t                 buffer_size;
    size_t*                p_buffer_size = &buffer_size;

    //
    // rocsparse_csritilu0_buffer_size.
    //
    {
#define PARAMS_BUFFER_SIZE \
    handle, alg, option, maxiter, m, nnz, ptr, ind, base, datatype, p_buffer_size

        static constexpr int nexcl       = 2;
        static constexpr int excl[nexcl] = {2, 3};
        auto_testing_bad_arg(rocsparse_csritilu0_buffer_size, nexcl, excl, PARAMS_BUFFER_SIZE);

        option                  = -1;
        rocsparse_status status = rocsparse_csritilu0_buffer_size(PARAMS_BUFFER_SIZE);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        option = 0;

        maxiter = -1;
        status  = rocsparse_csritilu0_buffer_size(PARAMS_BUFFER_SIZE);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        maxiter = 1;

#undef PARAMS_BUFFER_SIZE
    }

    //
    // rocsparse_csritilu0_preprocess.
    //
    {
        buffer_size  = 1;
        void* buffer = (void*)0x4;
#define PARAMS_PREPROCESS \
    handle, alg, option, maxiter, m, nnz, ptr, ind, base, datatype, buffer_size, buffer
        // 0     1      2      3      4    5   6    7    8       9        10           11

        static constexpr int nargs_to_exclude   = 3;
        static constexpr int args_to_exclude[3] = {2, 3, 10};

        auto_testing_bad_arg(
            rocsparse_csritilu0_preprocess, nargs_to_exclude, args_to_exclude, PARAMS_PREPROCESS);

        option                  = -1;
        rocsparse_status status = rocsparse_csritilu0_preprocess(PARAMS_PREPROCESS);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        option = 0;

        maxiter = -1;
        status  = rocsparse_csritilu0_preprocess(PARAMS_PREPROCESS);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        maxiter = 1;

#undef PARAMS_PREPROCESS
    }

    //
    // rocsparse_csritilu0_compute.
    //
    {
        buffer_size                  = 1;
        T*                 val       = (T*)0x4;
        T*                 ilu0      = (T*)0x4;
        void*              buffer    = (void*)0x4;
        floating_data_t<T> tol       = static_cast<floating_data_t<T>>(1.0e-8);
        rocsparse_int*     p_maxiter = (rocsparse_int*)0x4;

#define PARAMS_COMPUTE \
    handle, alg, option, p_maxiter, tol, m, nnz, ptr, ind, val, ilu0, base, buffer_size, buffer
        // 0     1      2      3         4   5   6    7    8    9    10    11    12             13

        static constexpr int nargs_to_exclude   = 3;
        static constexpr int args_to_exclude[3] = {2, 4, 12};

        auto_testing_bad_arg(
            rocsparse_csritilu0_compute<T>, nargs_to_exclude, args_to_exclude, PARAMS_COMPUTE);

        option                  = -1;
        rocsparse_status status = rocsparse_csritilu0_compute<T>(PARAMS_COMPUTE);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        option = 0;

        tol    = -1;
        status = rocsparse_csritilu0_compute<T>(PARAMS_COMPUTE);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value, status);
        tol = 1.0e-8;

#undef PARAMS_COMPUTE
    }
}

template <typename T>
struct csritilu0_params_t
{
    rocsparse_itilu0_alg alg;
    rocsparse_int        options;
    rocsparse_int        maxiter;
    floating_data_t<T>   tol;
    rocsparse_datatype   datatype;
    csritilu0_params_t() = delete;
    csritilu0_params_t(rocsparse_itilu0_alg alg_,
                       rocsparse_int        options_,
                       rocsparse_int        maxiter_,
                       floating_data_t<T>   tol_)
        : alg(alg_)
        , options(options_)
        , maxiter(maxiter_)
        , tol(tol_)
        , datatype(rocsparse_datatype_t::get<T>()){};
};

template <typename T>
void testing_csritilu0(const Arguments& arg)
{
    static constexpr bool          verbose   = false;
    static constexpr rocsparse_int s_maxiter = 1000;
    floating_data_t<T> tol = (sizeof(floating_data_t<T>) == sizeof(float)) ? 2e-7 : 5e-15;
    if(arg.numericboost)
    {
        tol = arg.boosttol;
    }
    rocsparse_int options = 0;

    options |= rocsparse_itilu0_option_stopping_criteria; // Add stopping criteria.
    options |= rocsparse_itilu0_option_compute_nrm_residual; // Compute the norm of the residual.
    // options |= rocsparse_itilu0_option_verbose; // Put verbose.
    // options |= rocsparse_itilu0_option_convergence_history; // Get convergence history.
    // options |= rocsparse_itilu0_option_compute_nrm_correction; // Compute the norm of the correction.
    // options |= rocsparse_itilu0_option_coo_format; // Use internal sparse coordinate format.

    csritilu0_params_t<T> p(arg.itilu0_alg, options, s_maxiter, tol);
    //
    // Set constant parameters.
    //
    const bool                  to_int    = false; // arg.timing ? false : true;
    static constexpr bool       full_rank = true;
    rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);

    // Create rocsparse handle
    rocsparse_local_handle handle;
    rocsparse_status       status;

    //
    // Initialize csr matrix.
    //
    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA);
    if(false)
    {
        if(hA.m > 0 && hA.nnz > 0)
        {
            rocsparse_host_csrscaling_ruiz<T, rocsparse_int, rocsparse_int>(
                hA.m, hA.n, hA.nnz, hA.ptr, hA.ind, hA.val, hA.base);
        }
    }

    //
    // Transfer matrix A to device.
    //
    device_csr_matrix<T> dA(hA);

    p.maxiter          = s_maxiter;
    size_t buffer_size = 0;
    void*  buffer{};

    status = rocsparse_csritilu0_buffer_size(handle,
                                             p.alg,
                                             p.options,
                                             p.maxiter,

                                             //
                                             dA.m,
                                             dA.nnz,
                                             dA.ptr,
                                             dA.ind,
                                             dA.base,

                                             p.datatype,
                                             &buffer_size);

    if(dA.nnz == 0 && dA.m > 0)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_zero_pivot, status);
        return;
    }
    else
    {
        CHECK_ROCSPARSE_ERROR(status);
    }
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size));
    rocsparse_status       status_buffer_size = status;
    device_dense_vector<T> ilu0(dA.nnz);
    if(arg.unit_check)
    {
        //
        // Compute CSRILU0 with gaussian elimination.
        //
        device_csr_matrix<T> dA_csrilu0(dA);
        int                  pivot = 0;
        rocsparse_status     status_csrilu0_buffer_size;
        rocsparse_status     status_csrilu0_analysis;
        rocsparse_status     status_csrilu0;
        {
            rocsparse_analysis_policy apol = rocsparse_analysis_policy_force;
            rocsparse_solve_policy    spol = arg.spol;
            rocsparse_local_mat_info  info;
            rocsparse_local_mat_descr descr;

            // Set matrix index base
            status = rocsparse_set_mat_index_base(descr, dA.base);
            if(status != rocsparse_status_success)
            {
                CHECK_ROCSPARSE_ERROR(status);
            }

            status = csrilu0(handle,
                             dA_csrilu0,
                             descr,
                             apol,
                             spol,
                             info,
                             &pivot,
                             status_csrilu0_buffer_size,
                             status_csrilu0_analysis,
                             status_csrilu0);

            if(status_csrilu0_buffer_size != rocsparse_status_success)
            {
                EXPECT_ROCSPARSE_STATUS(status_csrilu0_buffer_size, status_buffer_size);
                return;
            }

#ifndef NDEBUG
            std::cout << "status csrilu0 analysis : " << status_csrilu0_analysis << std::endl;
            std::cout << "status csrilu0 compute  : " << status_csrilu0 << std::endl;
#endif

            if(status != rocsparse_status_success && status != rocsparse_status_zero_pivot)
            {
                CHECK_ROCSPARSE_ERROR(status);
            }

            if(status_csrilu0 == rocsparse_status_zero_pivot)
            {
                std::cerr << "numerical pivot needed from rocsparse_csrilu0, skip the test."
                          << std::endl;
                return;
            }
        }

        p.maxiter = s_maxiter;
        status    = rocsparse_csritilu0_preprocess(handle,
                                                p.alg,
                                                p.options,
                                                p.maxiter,
                                                //
                                                dA.m,
                                                dA.nnz,
                                                dA.ptr,

                                                dA.ind,
                                                dA.base,
                                                p.datatype,
                                                buffer_size,
                                                buffer);

        //
        // Must be consistent with csrilu0_analysis, i.e. if a zero pivot is found.
        //
        EXPECT_ROCSPARSE_STATUS(status_csrilu0_analysis, status);
        if(status == rocsparse_status_zero_pivot)
        {
            //
            // If structural zero pivot is detected then we can't compute ILU0 with this version since nnz(LDU) > nnz(A).
            //
            return;
        }

        //
        // Update solution from the GAUSSIAN elimination, if there is no zero pivot it must work.
        //
        if(status_csrilu0 != rocsparse_status_zero_pivot)
        {
            p.maxiter = s_maxiter;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            status = rocsparse_csritilu0_compute<T>(handle,
                                                    p.alg,
                                                    p.options,

                                                    &p.maxiter,
                                                    p.tol,

                                                    dA.m,
                                                    dA.nnz,
                                                    dA.ptr,
                                                    dA.ind,
                                                    dA.val,
                                                    dA_csrilu0.val,
                                                    dA.base,

                                                    buffer_size,
                                                    buffer);
            CHECK_ROCSPARSE_ERROR(status);
            if(dA.m == 0 || dA.nnz == 0)
            {
                unit_check_scalar((rocsparse_int)0, p.maxiter);
            }
            else
            {
                if(p.maxiter != 1)
                {
                    if(verbose)
                    {
                        std::cerr
                            << "update the Gaussian elimination takes more than 1 iteration ( = "
                            << p.maxiter << " )" << std::endl;
                    }
                    if((p.options & rocsparse_itilu0_option_convergence_history) > 0)
                    {
                        floating_data_t<T>* history       = new floating_data_t<T>[p.maxiter * 2];
                        rocsparse_int       history_niter = 0;
                        status                            = rocsparse_csritilu0_history<T>(
                            handle, p.alg, &history_niter, history, buffer_size, buffer);

                        const bool nrm_corr
                            = (p.options & rocsparse_itilu0_option_compute_nrm_correction) > 0;
                        const bool nrm_residual
                            = (p.options & rocsparse_itilu0_option_compute_nrm_residual) > 0;
                        for(rocsparse_int i = 0; i < history_niter; ++i)
                        {
                            std::cout << std::setw(12) << i;
                            if(nrm_corr)
                            {
                                std::cout << std::setw(12) << history[i];
                            }
                            if(nrm_residual)
                            {
                                std::cout << std::setw(12) << history[history_niter + i];
                            }
                            std::cout << std::endl;
                        }
                        delete[] history;
                    }
                }

                // Let's observe.
                // unit_check_scalar((rocsparse_int)1,p.maxiter);
            }
        }

        //
        // Compute solution.
        //
        hipMemset((T*)ilu0, 0, sizeof(T) * dA.nnz);

        p.maxiter = s_maxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_csritilu0_compute<T>(handle,
                                                p.alg,
                                                p.options,

                                                &p.maxiter,
                                                p.tol,

                                                dA.m,
                                                dA.nnz,
                                                dA.ptr,
                                                dA.ind,
                                                dA.val,
                                                ilu0,
                                                dA.base,

                                                buffer_size,
                                                buffer);
        if(status == rocsparse_status_zero_pivot)
        {
            std::cout << "rocsparse_csritilu0_compute divergence " << std::endl;
        }

        EXPECT_ROCSPARSE_STATUS(status_csrilu0, status);
        CHECK_ROCSPARSE_ERROR(status);

        if(status_csrilu0 == rocsparse_status_zero_pivot)

        {
            std::cout << "rocsparse_csrilu0 divergence, return. " << std::endl;
            return;
        }

        if((p.options & rocsparse_itilu0_option_convergence_history) > 0)
        {
            floating_data_t<T>* history       = new floating_data_t<T>[p.maxiter * 2];
            rocsparse_int       history_niter = 0;
            status                            = rocsparse_csritilu0_history<T>(
                handle, p.alg, &history_niter, history, buffer_size, buffer);
            CHECK_ROCSPARSE_ERROR(status);

            const bool nrm_corr = (p.options & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool nrm_residual
                = (p.options & rocsparse_itilu0_option_compute_nrm_residual) > 0;

            for(rocsparse_int i = 0; i < history_niter; ++i)
            {
                std::cout << std::setw(12) << i;
                if(nrm_corr)
                {
                    std::cout << std::setw(12) << history[i];
                }
                if(nrm_residual)
                {
                    std::cout << std::setw(12) << history[history_niter + i];
                }
                std::cout << std::endl;
            }
            delete[] history;
        }
        if(pivot != -1)
        {
            std::cerr << "rocsparse_status_zero_pivot is detected, we don't compare" << std::endl;
        }
        else
        {
            if(sizeof(floating_data_t<T>) == sizeof(double))
            {
                ilu0.near_check(dA_csrilu0.val, 1.0e-5);
            }
            else
            {
                ilu0.near_check(dA_csrilu0.val);
            }
        }
    }

    if(arg.timing)
    {
        const int number_cold_calls = 2;
        const int number_hot_calls  = arg.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            dA.val.transfer_from(hA.val);
            p.maxiter = s_maxiter;

            status = rocsparse_csritilu0_preprocess(handle,
                                                    p.alg,
                                                    p.options,
                                                    p.maxiter,
                                                    dA.m,
                                                    dA.nnz,
                                                    dA.ptr,
                                                    dA.ind,
                                                    dA.base,

                                                    p.datatype,
                                                    buffer_size,
                                                    buffer);

            CHECK_ROCSPARSE_ERROR(status);
            status = rocsparse_csritilu0_compute<T>(handle,
                                                    p.alg,
                                                    p.options,

                                                    &p.maxiter,
                                                    p.tol,

                                                    dA.m,
                                                    dA.nnz,
                                                    dA.ptr,
                                                    dA.ind,
                                                    dA.val,
                                                    ilu0,
                                                    dA.base,

                                                    buffer_size,
                                                    buffer);
            CHECK_ROCSPARSE_ERROR(status);
        }

        double gpu_presolve_time_used = 0;
        double gpu_solve_time_used    = 0;
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            //
            // Reinitialize, otherwise it will stay converged.
            //
            hipMemset((T*)ilu0, 0, sizeof(T) * dA.nnz);

            p.maxiter                          = s_maxiter;
            double gpu_presolve_time_used_iter = get_time_us();
            status                             = rocsparse_csritilu0_preprocess(handle,
                                                    p.alg,
                                                    p.options,
                                                    p.maxiter,

                                                    dA.m,
                                                    dA.nnz,
                                                    dA.ptr,
                                                    dA.ind,
                                                    dA.base,

                                                    p.datatype,
                                                    buffer_size,
                                                    buffer);
            gpu_presolve_time_used_iter        = (get_time_us() - gpu_presolve_time_used_iter);
            gpu_presolve_time_used += gpu_presolve_time_used_iter;
            CHECK_ROCSPARSE_ERROR(status);
            //
            //
            //
            double gpu_solve_time_used_iter = get_time_us();
            p.maxiter                       = s_maxiter;

            status = rocsparse_csritilu0_compute<T>(handle,
                                                    p.alg,
                                                    p.options,

                                                    &p.maxiter,
                                                    p.tol,

                                                    dA.m,
                                                    dA.nnz,
                                                    dA.ptr,
                                                    dA.ind,
                                                    dA.val,
                                                    ilu0,
                                                    dA.base,

                                                    buffer_size,
                                                    buffer);

            gpu_solve_time_used_iter = (get_time_us() - gpu_solve_time_used_iter);
            gpu_solve_time_used += gpu_solve_time_used_iter;
            CHECK_ROCSPARSE_ERROR(status);
        }

        gpu_solve_time_used /= number_hot_calls;
        gpu_presolve_time_used /= number_hot_calls;

        //
        // gflops ?
        //
        display_timing_info("M",
                            dA.m,
                            "nnz",
                            dA.nnz,
                            "maxiter",
                            p.maxiter,
                            "tol",
                            p.tol,
                            "presolve",
                            get_gpu_time_msec(gpu_presolve_time_used),
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_solve_time_used),
                            "buffer_size",
                            buffer_size);
    }
    CHECK_HIP_ERROR(hipFree(buffer));
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_csritilu0_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csritilu0<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
