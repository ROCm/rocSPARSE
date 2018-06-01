/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_GTHRZ_HPP
#define ROCSPARSE_GTHRZ_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "gthrz_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_gthrz_template(rocsparse_handle handle,
                                          rocsparse_int nnz,
                                          const T* y,
                                          T* x_val,
                                          const rocsparse_int* x_ind,
                                          rocsparse_index_base idx_base)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_GTHRZ_HPP
