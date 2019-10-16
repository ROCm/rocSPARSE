/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "handle.h"
#include "definitions.h"
#include "logging.h"

#include <hip/hip_runtime.h>

__global__ void init_kernel(){};

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

    // Obtain size for coomv device buffer
    rocsparse_int nthreads = properties.maxThreadsPerBlock;
    rocsparse_int nprocs   = properties.multiProcessorCount;
    rocsparse_int nblocks  = (nprocs * nthreads - 1) / 128 + 1;
    rocsparse_int nwfs     = nblocks * (128 / properties.warpSize);

    size_t coomv_size = (((sizeof(rocsparse_int) + 16) * nwfs - 1) / 256 + 1) * 256;

    // Allocate device buffer
    buffer_size = (coomv_size > 1024 * 1024) ? coomv_size : 1024 * 1024;
    THROW_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Device one
    THROW_IF_HIP_ERROR(hipMalloc(&sone, sizeof(float)));
    THROW_IF_HIP_ERROR(hipMalloc(&done, sizeof(double)));
    THROW_IF_HIP_ERROR(hipMalloc(&cone, sizeof(rocsparse_float_complex)));
    THROW_IF_HIP_ERROR(hipMalloc(&zone, sizeof(rocsparse_double_complex)));

    // Execute empty kernel for initialization
    hipLaunchKernelGGL(init_kernel, dim3(1), dim3(1), 0, stream);

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync(sone, 0, sizeof(float), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(done, 0, sizeof(double), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(cone, 0, sizeof(rocsparse_float_complex), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(zone, 0, sizeof(rocsparse_double_complex), stream));

    float  hsone = 1.0f;
    double hdone = 1.0;

    rocsparse_float_complex  hcone = rocsparse_float_complex(1.0f, 0.0f);
    rocsparse_double_complex hzone = rocsparse_double_complex(1.0, 0.0);

    THROW_IF_HIP_ERROR(hipMemcpyAsync(sone, &hsone, sizeof(float), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(done, &hdone, sizeof(double), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        cone, &hcone, sizeof(rocsparse_float_complex), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        zone, &hzone, sizeof(rocsparse_double_complex), hipMemcpyHostToDevice, stream));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

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
    PRINT_IF_HIP_ERROR(hipFree(buffer));
    PRINT_IF_HIP_ERROR(hipFree(sone));
    PRINT_IF_HIP_ERROR(hipFree(done));
    PRINT_IF_HIP_ERROR(hipFree(cone));
    PRINT_IF_HIP_ERROR(hipFree(zone));

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
 * using rocsparse_destroy_csrmv_info().
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

/********************************************************************************
 * \brief rocsparse_csrtr_info is a structure holding the rocsparse csrsv and
 * csrilu0 data gathered during csrsv_analysis and csrilu0_analysis. It must be
 * initialized using the rocsparse_create_csrtr_info() routine. It should be
 * destroyed at the end using rocsparse_destroy_csrtr_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrtr_info(rocsparse_csrtr_info* info)
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
            *info = new _rocsparse_csrtr_info;
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
rocsparse_status rocsparse_destroy_csrtr_info(rocsparse_csrtr_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    // Clean up
    if(info->row_map != nullptr)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->row_map));
        info->row_map = nullptr;
    }

    if(info->csr_diag_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->csr_diag_ind));
        info->csr_diag_ind = nullptr;
    }

    if(info->zero_pivot != nullptr)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->zero_pivot));
        info->zero_pivot = nullptr;
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

/********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the rocsparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using rocsparse_destroy_csrgemm_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrgemm_info(rocsparse_csrgemm_info* info)
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
            *info = new _rocsparse_csrgemm_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Destroy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrgemm_info(rocsparse_csrgemm_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
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
