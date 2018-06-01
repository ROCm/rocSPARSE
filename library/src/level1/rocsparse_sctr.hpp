/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_SCTR_HPP
#define ROCSPARSE_SCTR_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "sctr_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_sctr_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
                                         const T* x_val,
                                         const rocsparse_int* x_ind,
                                         T* y,
                                         rocsparse_index_base idx_base)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_SCTR_HPP
