/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "handle.h"
#include "logging.h"

#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocsparse_handle::_rocsparse_handle()
{
    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));

    // Device wavefront size
    wavefront_size = properties.warpSize;

    // Layer mode
    char* str_layer_mode;
    if((str_layer_mode = getenv("ROCSPARSE_LAYER")) == NULL)
    {
        layer_mode = rocsparse_layer_mode_none;
    }
    else
    {
        layer_mode = (rocsparse_layer_mode)(atoi(str_layer_mode));
    }

    // Open log file
    if(layer_mode & rocsparse_layer_mode_log_trace)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSE_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if(layer_mode & rocsparse_layer_mode_log_bench)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSE_LOG_BENCH_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocsparse_handle::~_rocsparse_handle()
{
    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
}

/*******************************************************************************
 * Exactly like cuSPARSE, rocSPARSE only uses one stream for one API routine
 ******************************************************************************/

/*******************************************************************************
 * set stream:
   This API assumes user has already created a valid stream
   Associate the following rocsparse API call with this user provided stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::set_stream(hipStream_t user_stream)
{
    // TODO check if stream is valid
    stream = user_stream;
    return rocsparse_status_success;
}

/*******************************************************************************
 * get stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::get_stream(hipStream_t* user_stream) const
{
    *user_stream = stream;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * rocsparse_create_csrmv_info() routine. It should be destroyed at the end
 * rocsparse_destroy_csrmv_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrmv_info(rocsparse_csrmv_info* info)
{
    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *info = new _rocsparse_csrmv_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrmv_info(rocsparse_csrmv_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    // Clean up row blocks
    if(info->size > 0)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->row_blocks));
    }

    // Destruct
    try
    {
        delete info;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}
