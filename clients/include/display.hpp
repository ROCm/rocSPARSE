/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief display.hpp provides common testing utilities.
 */

#pragma once
#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "rocsparse_test.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <vector>

struct display_key_t
{
    //
    // Enumerate keys.
    //
    typedef enum enum_
    {
        trans_A = 0,
        trans_B,
        nnz_A,
        nnz_B,
        nnz_C,
        nnz_D,
        iters,
        function,
        ctype,
        itype,
        jtype,
        alpha,
        beta,
        gflops,
        bandwidth,
        time_ms,
        algorithm,
        M,
        N,
        K,
        dir_A,
        nnzb_A,
        bdir_A,
        bdim_A,
        analysis_ms,
        diag_type,
        fill_mode,
        analysis_policy,
        solve_policy,
        permute,

        bdir  = bdir_A,
        bdim  = bdim_A,
        dir   = dir_A,
        nnzb  = nnzb_A,
        trans = trans_A,
        nnz   = nnz_A,
    } key_t;

    static const char* to_str(key_t key_)
    {
        switch(key_)
        {
        case gflops:
        {
            return s_timing_info_perf;
        }

        case diag_type:
        {
            return "diag_type";
        }

        case permute:
        {
            return "permute";
        }

        case analysis_ms:
        {
            return "analysis";
        }

        case fill_mode:
        {
            return "fill_mode";
        }

        case analysis_policy:
        {
            return "analysis_policy";
        }

        case solve_policy:
        {
            return "solve_policy";
        }

        case algorithm:
        {
            return "algorithm";
        }

        case M:
        {
            return "M";
        }

        case bdir_A:
        {
            return "bdir_A";
        }

        case dir_A:
        {
            return "dir_A";
        }

        case bdim_A:
        {
            return "bdim_A";
        }

        case N:
        {
            return "N";
        }

        case K:
        {
            return "K";
        }

        case bandwidth:
        {
            return s_timing_info_bandwidth;
        }

        case time_ms:
        {
            return s_timing_info_time;
        }

        case alpha:
        {
            return "alpha";
        }
        case beta:
        {
            return "beta";
        }
        case trans_A:
        {
            return "transA";
        }
        case trans_B:
        {
            return "transB";
        }

        case nnzb_A:
        {
            return "nnzb_A";
        }

        case nnz_A:
        {
            return "nnz_A";
        }
        case nnz_B:
        {
            return "nnz_B";
        }
        case nnz_C:
        {
            return "nnz_C";
        }
        case nnz_D:
        {
            return "nnz_D";
        }
        case iters:
        {
            return "iter";
        }
        case function:
        {
            return "function";
        }
        case ctype:
        {
            return "ctype";
        }

        case itype:
        {
            return "itype";
        }
        case jtype:
        {
            return "jtype";
        }
        }
    }
};

template <typename S>
inline const char* display_to_string(S s)
{
    return s;
};
template <>
inline const char* display_to_string(display_key_t::key_t s)
{
    return display_key_t::to_str(s);
};

//
// Template to display timing information.
//
template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, S name, T t)
{
    out << std::setw(n) << display_to_string(name);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, S name, T t, Ts... ts)
{
    out << std::setw(n) << display_to_string(name);
    display_timing_info_legend(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, S name, T t)
{
    out << std::setw(n) << t;
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, S name, T t, Ts... ts)
{
    out << std::setw(n) << t;
    display_timing_info_values(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend_noresults(std::ostream& out, int n, S name_, T t)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend_noresults(std::ostream& out, int n, S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
    display_timing_info_legend_noresults(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values_noresults(std::ostream& out, int n, S name_, T t)
{
    const char* name = display_to_string(name_);

    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values_noresults(std::ostream& out, int n, S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
    display_timing_info_values_noresults(out, n, ts...);
}

template <typename T>
inline void grab_results(double values[3], display_key_t::key_t key, T t)
{
}

template <>
inline void grab_results<double>(double values[3], display_key_t::key_t key, double t)
{
    const char* name = display_to_string(key);
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

template <typename S, typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], S name, T t)
{
    grab_results(values, name, t);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], S name, T t, Ts... ts)
{
    grab_results(values, name, t);
    display_timing_info_grab_results(values, ts...);
}

bool display_timing_info_is_stdout_disabled();

template <typename S, typename T, typename... Ts>
inline void display_timing_info_generate(std::ostream& out, int n, S name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    rocsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values(out, n, name, t, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_generate_params(std::ostream& out, int n, S name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    rocsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values_noresults(out, n, name, t, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], S name_, T t)
{
    const char* name = display_to_string(name_);
    int         len  = strlen(name);
    mx[0]            = std::max(len, mx[0]);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    int         len  = strlen(name);
    mx[0]            = std::max(len, mx[0]);
    display_timing_info_max_size_strings(mx, ts...);
}

template <typename S, typename... Ts>
inline void display_timing_info_main(S name, Ts... ts)
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
                                 display_key_t::iters,                                       \
                                 arg.iters,                                                  \
                                 "verified",                                                 \
                                 (arg.unit_check ? "yes" : "no"),                            \
                                 display_key_t::function,                                    \
                                 &arg.function[0],                                           \
                                 "import",                                                   \
                                 importname,                                                 \
                                 display_key_t::ctype,                                       \
                                 ctypename,                                                  \
                                 display_key_t::itype,                                       \
                                 itypename,                                                  \
                                 display_key_t::jtype,                                       \
                                 jtypename);                                                 \
    } while(false)

#endif // DISPLAY_HPP
