/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "matrix.h"
#include "rocsparse.h"



/********************************************************************************
 * \brief rocsparseCreateMatDescr_t is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparseCreateMatDescr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparseDestroyMatDescr().
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseCreateMatDescr(rocsparseMatDescr_t *descrA)
{
    if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        // Allocate
        try
        {
            *descrA = new rocsparseMatDescr;
        }
        catch(rocsparseStatus_t status)
        {
            return status;
        }
        return ROCSPARSE_STATUS_SUCCESS;
    }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseDestroyMatDescr(rocsparseMatDescr_t descrA)
{
    // Destruct
    try
    {
        delete descrA;
    }
    catch(rocsparseStatus_t status)
    {
        return status;
    }
    return ROCSPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseSetMatIndexBase(rocsparseMatDescr_t descrA,
                                           rocsparseIndexBase_t base)
{
    // Check if descriptor is valid
    if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    if (base != ROCSPARSE_INDEX_BASE_ZERO &&
        base != ROCSPARSE_INDEX_BASE_ONE)
    {
        return ROCSPARSE_STATUS_INVALID_VALUE;
    }
    descrA->base = base;
    return ROCSPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
extern "C"
rocsparseIndexBase_t rocsparseGetMatIndexBase(const rocsparseMatDescr_t descrA)
{
    // If descriptor is invalid, default index base is returned
    if (descrA == nullptr)
    {
        return ROCSPARSE_INDEX_BASE_ZERO;
    }
    return descrA->base;
}

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
extern "C"
rocsparseStatus_t rocsparseSetMatType(rocsparseMatDescr_t descrA,
                                      rocsparseMatrixType_t type)
{
    // Check if descriptor is valid
    if (descrA == nullptr)
    {
        return ROCSPARSE_STATUS_INVALID_POINTER;
    }
    if (type != ROCSPARSE_MATRIX_TYPE_GENERAL &&
        type != ROCSPARSE_MATRIX_TYPE_SYMMETRIC &&
        type != ROCSPARSE_MATRIX_TYPE_HERMITIAN)
    {
        return ROCSPARSE_STATUS_INVALID_VALUE;
    }
    descrA->type = type;
    return ROCSPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
extern "C"
rocsparseMatrixType_t rocsparseGetMatType(const rocsparseMatDescr_t descrA)
{
    // If descriptor is invalid, default matrix type is returned
    if (descrA == nullptr)
    {
        return ROCSPARSE_MATRIX_TYPE_GENERAL;
    }
    return descrA->type;
}
