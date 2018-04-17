/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_STATUS_H_
#define ROCSPARSE_STATUS_H_

#include "rocsparse.h"

#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 ******************************************************************************/
rocsparseStatus_t get_rocsparse_status_for_hip_status(hipError_t status);

#endif // ROCSPARSE_STATUS_H_
