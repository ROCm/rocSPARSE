/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief auto_testing_bad_arg.hpp provides common testing utilities.
 */

#pragma once
#ifndef AUTO_TESTING_BAD_ARG_HPP
#define AUTO_TESTING_BAD_ARG_HPP

#include "rocsparse_test.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <vector>
//
// PROVIDE TEMPLATE FOR AUTO TESTING BAD ARGUMENTS
//

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T& p);

template <typename T>
inline void auto_testing_bad_arg_set_invalid(T*& p)
{
    p = nullptr;
}

template <>
inline void auto_testing_bad_arg_set_invalid(int32_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(int64_t& p)
{
    p = -1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(float& p)
{
    p = static_cast<float>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(double& p)
{
    p = static_cast<double>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_float_complex& p)
{
    p = static_cast<rocsparse_float_complex>(-1);
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_double_complex& p)
{
    p = static_cast<rocsparse_double_complex>(-1);
}

template <typename T>
inline rocsparse_status auto_testing_bad_arg_get_status(T& p)
{
    return rocsparse_status_invalid_pointer;
}

template <typename T>
inline rocsparse_status auto_testing_bad_arg_get_status(const T& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_handle& p)
{
    return rocsparse_status_invalid_handle;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmat_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spvec_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dnmat_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dnvec_descr& p)
{
    return rocsparse_status_invalid_pointer;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(int32_t& p)
{
    return rocsparse_status_invalid_size;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(int64_t& p)
{
    return rocsparse_status_invalid_size;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_operation& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_order& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_index_base& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_matrix_type& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_fill_mode& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_storage_mode& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_gtsv_interleaved_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sparse_to_dense_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_dense_to_sparse_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsv_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sddmm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_gpsv_interleaved_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgemm_alg& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_indextype& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_datatype& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_analysis_policy& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_solve_policy& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_direction& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_action& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_hyb_partition& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsv_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spsm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spmm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_spgemm_stage& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_operation& p)
{
    p = (rocsparse_operation)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_order& p)
{
    p = (rocsparse_order)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_index_base& p)
{
    p = (rocsparse_index_base)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_matrix_type& p)
{
    p = (rocsparse_matrix_type)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_fill_mode& p)
{
    p = (rocsparse_fill_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_storage_mode& p)
{
    p = (rocsparse_storage_mode)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_gtsv_interleaved_alg& p)
{
    p = (rocsparse_gtsv_interleaved_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sparse_to_dense_alg& p)
{
    p = (rocsparse_sparse_to_dense_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_dense_to_sparse_alg& p)
{
    p = (rocsparse_dense_to_sparse_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmv_alg& p)
{
    p = (rocsparse_spmv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsv_alg& p)
{
    p = (rocsparse_spsv_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsm_alg& p)
{
    p = (rocsparse_spsm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sddmm_alg& p)
{
    p = (rocsparse_sddmm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmm_alg& p)
{
    p = (rocsparse_spmm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_gpsv_interleaved_alg& p)
{
    p = (rocsparse_gpsv_interleaved_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgemm_alg& p)
{
    p = (rocsparse_spgemm_alg)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_indextype& p)
{
    p = (rocsparse_indextype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_datatype& p)
{
    p = (rocsparse_datatype)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_analysis_policy& p)
{
    p = (rocsparse_analysis_policy)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_direction& p)
{
    p = (rocsparse_direction)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_action& p)
{
    p = (rocsparse_action)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_hyb_partition& p)
{
    p = (rocsparse_hyb_partition)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_solve_policy& p)
{
    p = (rocsparse_solve_policy)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsv_stage& p)
{
    p = (rocsparse_spsv_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spsm_stage& p)
{
    p = (rocsparse_spsm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spmm_stage& p)
{
    p = (rocsparse_spmm_stage)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_spgemm_stage& p)
{
    p = (rocsparse_spgemm_stage)-1;
}

template <typename... T>
struct auto_testing_bad_arg_t
{
    inline auto_testing_bad_arg_t(){};
    inline auto_testing_bad_arg_t(int current, int ith, rocsparse_status& status){};
};

template <typename T, typename... Rest>
struct auto_testing_bad_arg_t<T, Rest...>
{
    inline auto_testing_bad_arg_t(T first, Rest... rest)
        : first(first)
        , rest(rest...)
    {
    }

    inline auto_testing_bad_arg_t(int ith, rocsparse_status& status, T& first, Rest&... rest)
        : auto_testing_bad_arg_t(0, ith, status, first, rest...)
    {
    }

    inline auto_testing_bad_arg_t(
        int current, int ith, rocsparse_status& status, T& first, Rest&... rest)
        : first(first)
        , rest(current + 1, ith, status, rest...)
    {
        if(current == ith)
        {
            status = auto_testing_bad_arg_get_status<T>(first);
            auto_testing_bad_arg_set_invalid(this->first);
        }
    }

    T                               first;
    auto_testing_bad_arg_t<Rest...> rest;
};

template <typename C, typename T>
inline void auto_testing_bad_arg_copy(const C& data, T& t)
{
    t = data.first;
}

template <typename C, typename T, typename... Ts>
inline void auto_testing_bad_arg_copy(const C& data, T& t, Ts&... ts)
{
    t = data.first;
    auto_testing_bad_arg_copy(data.rest, ts...);
}

template <typename T>
inline void auto_testing_bad_arg_print(T& t)
{
    std::cout << " " << t << "," << std::endl;
}

template <typename T, typename... Ts>
inline void auto_testing_bad_arg_print(T& t, Ts&... ts)
{
    std::cout << " " << t << "," << std::endl;
    auto_testing_bad_arg_print(ts...);
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, Ts... ts)
{
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        auto_testing_bad_arg_t<Ts...> arguments(ts...);

        {
            rocsparse_status              status;
            auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
            auto_testing_bad_arg_copy(invalid_data, ts...);

            if(status != f(ts...))
            {
                std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                          << std::endl;
                auto_testing_bad_arg_print(ts...);
                EXPECT_ROCSPARSE_STATUS(f(ts...), status);
            }
        }

        auto_testing_bad_arg_copy(arguments, ts...);
    }
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, int n, const int* idx, Ts... ts)
{
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        bool exclude = false;
        for(int i = 0; i < n; ++i)
        {
            if(idx[i] == iarg)
            {
                exclude = true;
                break;
            }
        }

        if(!exclude)
        {
            auto_testing_bad_arg_t<Ts...> arguments(ts...);

            {
                rocsparse_status              status = rocsparse_status_success;
                auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
                auto_testing_bad_arg_copy(invalid_data, ts...);

                if(status != f(ts...))
                {
                    std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                              << std::endl;
                    auto_testing_bad_arg_print(ts...);
                    EXPECT_ROCSPARSE_STATUS(f(ts...), status);
                }
            }

            auto_testing_bad_arg_copy(arguments, ts...);
        }
    }
}

//
// Template to display timing information.
//
template <typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, const char* name, T t)
{
    out << std::setw(n) << name;
}

template <typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    out << std::setw(n) << name;
    display_timing_info_legend(out, n, ts...);
}

template <typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, const char* name, T t)
{
    out << std::setw(n) << t;
}

template <typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    out << std::setw(n) << t;
    display_timing_info_values(out, n, ts...);
}

template <typename T, typename... Ts>
inline void display_timing_info_legend_noresults(std::ostream& out, int n, const char* name, T t)
{
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
}

template <typename T, typename... Ts>
inline void
    display_timing_info_legend_noresults(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
    display_timing_info_legend_noresults(out, n, ts...);
}

template <typename T, typename... Ts>
inline void display_timing_info_values_noresults(std::ostream& out, int n, const char* name, T t)
{
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
}

template <typename T, typename... Ts>
inline void
    display_timing_info_values_noresults(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
    display_timing_info_values_noresults(out, n, ts...);
}

template <typename T>
inline void grab_results(double values[3], const char* name, T t)
{
}

template <>
inline void grab_results<double>(double values[3], const char* name, double t)
{
    if(!strcmp(name, s_timing_info_perf))
    {
        values[1] = t;
    }
    else if(!strcmp(name, s_timing_info_bandwidth))
    {
        values[2] = t;
    }
    else if(!strcmp(name, s_timing_info_time))
    {
        values[0] = t;
    }
}

template <typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], const char* name, T t)
{
    grab_results(values, name, t);
}

template <typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], const char* name, T t, Ts... ts)
{
    grab_results(values, name, t);
    display_timing_info_grab_results(values, ts...);
}

bool display_timing_info_is_stdout_disabled();

template <typename T, typename... Ts>
inline void display_timing_info_generate(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    rocsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values(out, n, name, t, ts...);
}

template <typename T, typename... Ts>
inline void
    display_timing_info_generate_params(std::ostream& out, int n, const char* name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    rocsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values_noresults(out, n, name, t, ts...);
}

template <typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], const char* name, T t)
{
    int len = strlen(name);
    mx[0]   = std::max(len, mx[0]);
}

template <typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], const char* name, T t, Ts... ts)
{
    int len = strlen(name);
    mx[0]   = std::max(len, mx[0]);
    display_timing_info_max_size_strings(mx, ts...);
}

template <typename... Ts>
inline void display_timing_info_main(const char* name, Ts... ts)
{
    //
    // To configure the size of std::setw.
    //
    int n = 0;
    display_timing_info_max_size_strings(&n, name, ts...);

    //
    //
    //
    n += 4;

    //
    // Legend
    //
    {
        std::ostringstream out_legend;
        out_legend.precision(2);
        out_legend.setf(std::ios::fixed);
        out_legend.setf(std::ios::left);
        if(!display_timing_info_is_stdout_disabled())
        {
            display_timing_info_legend(out_legend, n, name, ts...);
            std::cout << out_legend.str() << std::endl;
        }
        else
        {
            // store the string.
            display_timing_info_legend_noresults(out_legend, n, name, ts...);
            rocsparse_record_output_legend(out_legend.str());
        }
    }

    std::ostringstream out;
    out.precision(2);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);
    if(!display_timing_info_is_stdout_disabled())
    {
        display_timing_info_generate(out, n, name, ts...);
        std::cout << out.str() << std::endl;
    }
    else
    {
        display_timing_info_generate_params(out, n, name, ts...);
        // store the string.
        rocsparse_record_output(out.str());
    }
}

inline void rocsparse_get_matrixname(const char* f, char* name)
{
    int n = 0;
    while(f[n] != '\0')
        ++n;
    int cdir = 0;
    for(int i = 0; i < n; ++i)
    {
        if(f[i] == '/' || f[i] == '\\')
        {
            cdir = i + 1;
        }
    }
    int ddir = cdir;
    for(int i = cdir; i < n; ++i)
    {
        if(f[i] == '.')
        {
            ddir = i;
        }
    }

    if(ddir == cdir)
    {
        ddir = n;
    }

    for(int i = cdir; i < ddir; ++i)
    {
        name[i - cdir] = f[i];
    }
    name[ddir - cdir] = '\0';
}

#define display_timing_info(...)                                                             \
    do                                                                                       \
    {                                                                                        \
        char matrixname[64];                                                                 \
        if(rocsparse_arguments_has_datafile(arg))                                            \
            rocsparse_get_matrixname(&arg.filename[0], &matrixname[0]);                      \
        const char* importname                                                               \
            = (rocsparse_arguments_has_datafile(arg) ? &matrixname[0]                        \
                                                     : rocsparse_matrix2string(arg.matrix)); \
        const char* ctypename = rocsparse_datatype2string(arg.compute_type);                 \
        const char* itypename = rocsparse_indextype2string(arg.index_type_I);                \
        const char* jtypename = rocsparse_indextype2string(arg.index_type_J);                \
                                                                                             \
        display_timing_info_main(__VA_ARGS__,                                                \
                                 "iter",                                                     \
                                 arg.iters,                                                  \
                                 "verified",                                                 \
                                 (arg.unit_check ? "yes" : "no"),                            \
                                 "function",                                                 \
                                 &arg.function[0],                                           \
                                 "import",                                                   \
                                 importname,                                                 \
                                 "ctype",                                                    \
                                 ctypename,                                                  \
                                 "itype",                                                    \
                                 itypename,                                                  \
                                 "jtype",                                                    \
                                 jtypename);                                                 \
    } while(false)

//
// Compute gflops
//

inline double get_gpu_gflops(double gpu_time_used, double gflop_count)
{
    return gflop_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gflops(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gflops(gpu_time_used, count(ts...));
}

//
// Compute gbyte
//
inline double get_gpu_gbyte(double gpu_time_used, double gbyte_count)
{
    return gbyte_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gbyte(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gbyte(gpu_time_used, count(ts...));
}

inline double get_gpu_time_msec(double gpu_time_used)
{
    return gpu_time_used / 1e3;
}

#endif // AUTO_TESTING_BAD_ARG_HPP
