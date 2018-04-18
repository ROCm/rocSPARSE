/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_MATRIX_H_
#define ROCSPARSE_MATRIX_H_

#include "rocsparse.h"

struct rocsparseMatDescr
{
    // Matrix type
    rocsparseMatrixType_t type = ROCSPARSE_MATRIX_TYPE_GENERAL;
    // Fill mode TODO
//    rocsparseFillMode_t fill;
    // Diagonal type
//    rocsparseDiagType_t diag;
    // Index base
    rocsparseIndexBase_t base = ROCSPARSE_INDEX_BASE_ZERO;
};

#endif // ROCSPARSE_MATRIX_H_
