/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
 *  \brief utility.hpp provides common utilities
 */

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "rocsparse_test.hpp"

#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <string>
#include <vector>

/* ==================================================================================== */
/*! \brief  local handle which is automatically created and destroyed  */
class rocsparse_local_handle
{
    rocsparse_handle handle;

public:
    rocsparse_local_handle()
    {
        rocsparse_create_handle(&handle);
    }
    ~rocsparse_local_handle()
    {
        rocsparse_destroy_handle(handle);
    }

    // Allow rocsparse_local_handle to be used anywhere rocsparse_handle is expected
    operator rocsparse_handle&()
    {
        return handle;
    }
    operator const rocsparse_handle&() const
    {
        return handle;
    }
};

/* ==================================================================================== */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class rocsparse_local_mat_descr
{
    rocsparse_mat_descr descr;

public:
    rocsparse_local_mat_descr()
    {
        rocsparse_create_mat_descr(&descr);
    }
    ~rocsparse_local_mat_descr()
    {
        rocsparse_destroy_mat_descr(descr);
    }

    // Allow rocsparse_local_mat_descr to be used anywhere rocsparse_mat_descr is expected
    operator rocsparse_mat_descr&()
    {
        return descr;
    }
    operator const rocsparse_mat_descr&() const
    {
        return descr;
    }
};

/* ==================================================================================== */
/*! \brief  local matrix info which is automatically created and destroyed  */
class rocsparse_local_mat_info
{
    rocsparse_mat_info info;

public:
    rocsparse_local_mat_info()
    {
        rocsparse_create_mat_info(&info);
    }
    ~rocsparse_local_mat_info()
    {
        rocsparse_destroy_mat_info(info);
    }

    // Allow rocsparse_local_mat_info to be used anywhere rocsparse_mat_info is expected
    operator rocsparse_mat_info&()
    {
        return info;
    }
    operator const rocsparse_mat_info&() const
    {
        return info;
    }
};

/* ==================================================================================== */
/*! \brief  hyb matrix structure helper to access data for tests  */
struct test_hyb
{
    rocsparse_int           m;
    rocsparse_int           n;
    rocsparse_hyb_partition partition;
    rocsparse_int           ell_nnz;
    rocsparse_int           ell_width;
    rocsparse_int*          ell_col_ind;
    void*                   ell_val;
    rocsparse_int           coo_nnz;
    rocsparse_int*          coo_row_ind;
    rocsparse_int*          coo_col_ind;
    void*                   coo_val;
};

/* ==================================================================================== */
/*! \brief  local hyb matrix structure which is automatically created and destroyed  */
class rocsparse_local_hyb_mat
{
    rocsparse_hyb_mat hyb;

public:
    rocsparse_local_hyb_mat()
    {
        rocsparse_create_hyb_mat(&hyb);
    }
    ~rocsparse_local_hyb_mat()
    {
        rocsparse_destroy_hyb_mat(hyb);
    }

    // Allow rocsparse_local_hyb_mat to be used anywhere rocsparse_hyb_mat is expected
    operator rocsparse_hyb_mat&()
    {
        return hyb;
    }
    operator const rocsparse_hyb_mat&() const
    {
        return hyb;
    }
};

/* ==================================================================================== */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return
 *  wall time
 */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return
 *  wall time
 */
double get_time_us_sync(hipStream_t stream);

/* ==================================================================================== */
// Return path of this executable
std::string rocsparse_exepath();

#endif // UTILITY_HPP
