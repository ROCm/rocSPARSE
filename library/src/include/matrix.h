/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_MATRIX_H_
#define ROCSPARSE_MATRIX_H_

#include "rocsparse.h"

struct rocsparseMatDescr
{
    // Constructor
    rocsparseMatDescr();
    // Destructor
    ~rocsparseMatDescr();

    // Matrix index base
    rocsparseIndexBase_t base;
    // Matrix type
    rocsparseMatrixType_t type;
};

#endif // ROCSPARSE_MATRIX_H_
