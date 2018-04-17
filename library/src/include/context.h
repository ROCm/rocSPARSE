/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_CONTEXT_H_
#define ROCSPARSE_CONTEXT_H_

#include "rocsparse.h"

#include <iostream>
#include <fstream>
#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief rocsparseContext is a structure holding the rocsparse library context.
******************************************************************************/
struct rocsparseContext
{
    rocsparseContext();
    ~rocsparseContext();

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device warp size
    int warp_size;
    // stream
    hipStream_t stream;
    // pointer mode
    rocsparsePointerMode_t pointer_mode;
    // logging mode
    rocsparseLayerMode_t layer_mode;

    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ostream *log_trace_os;
    std::ostream *log_bench_os;
};

#endif // ROCSPARSE_CONTEXT_H_
