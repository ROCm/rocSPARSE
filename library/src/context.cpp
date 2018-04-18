/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "definitions.h"
#include "context.h"
#include "rocsparse.h"
#include "utility.h"

rocsparseContext::rocsparseContext()
{
    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));
    // Default is system stream
    stream = 0;
    // Default pointer mode is host
    pointer_mode = ROCSPARSE_POINTER_MODE_HOST;
    // Device warp size
    warp_size = properties.warpSize;

    // Layer mode
    char *str_layer_mode;
    if ((str_layer_mode = getenv("ROCSPARSE_LAYER")) == NULL)
    {
        layer_mode = ROCSPARSE_LAYER_MODE_NONE;
    }
    else
    {
        layer_mode = (rocsparseLayerMode_t) (atoi(str_layer_mode));
    }

    // Open log file
    if (layer_mode & ROCSPARSE_LAYER_MODE_LOG_TRACE)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSE_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if (layer_mode & ROCSPARSE_LAYER_MODE_LOG_BENCH)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSE_LOG_BENCH_PATH");
    }
}

rocsparseContext::~rocsparseContext()
{
    if (log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if (log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
}

rocsparseStatus_t rocsparseContext::setStream(hipStream_t streamId)
{
    // TODO check if stream is valid
    stream = streamId;
    return ROCSPARSE_STATUS_SUCCESS;
}

rocsparseStatus_t rocsparseContext::getStream(hipStream_t *streamId) const
{
    *streamId = stream;
    return ROCSPARSE_STATUS_SUCCESS;
}
