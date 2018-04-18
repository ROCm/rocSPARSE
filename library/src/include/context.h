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
    // Constructor
    rocsparseContext();
    // Destructor
    ~rocsparseContext();

    // Set stream
    rocsparseStatus_t setStream(hipStream_t streamId);
    // Get stream
    rocsparseStatus_t getStream(hipStream_t *streamId) const;

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device warp size
    int warp_size;
    // stream ; default stream is system stream NULL
    hipStream_t stream = 0;
    // pointer mode ; default mode is host
    rocsparsePointerMode_t pointer_mode = ROCSPARSE_POINTER_MODE_HOST;
    // logging mode
    rocsparseLayerMode_t layer_mode;

    // logging streams
    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ostream *log_trace_os;
    std::ostream *log_bench_os;
};

#endif // ROCSPARSE_CONTEXT_H_
