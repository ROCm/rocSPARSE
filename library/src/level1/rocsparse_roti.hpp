/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_ROTI_HPP
#define ROCSPARSE_ROTI_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "roti_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_roti_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
                                         T* x_val,
                                         const rocsparse_int* x_ind,
                                         T* y,
                                         const T* c,
                                         const T* s,
                                         rocsparse_index_base idx_base)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_ROTI_HPP
