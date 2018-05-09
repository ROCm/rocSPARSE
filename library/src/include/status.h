/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef STATUS_H
#define STATUS_H

#include "rocsparse.h"

#include <hip/hip_runtime_api.h>

/*******************************************************************************
 * \brief convert hipError_t to rocsparse_status
 ******************************************************************************/
rocsparse_status get_rocsparse_status_for_hip_status(hipError_t status);

#endif // STATUS_H
