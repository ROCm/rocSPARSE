/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSPARSE_UTILS_H_
#define ROCSPARSE_UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

template <typename T>
inline int gen2DLaplacianUS(int ndim, int **rowptr, int **col, T **val)
{

    int n = ndim * ndim;
    int nnz_mat = n * 5 - ndim * 4;

    *rowptr = (int*) malloc((n+1)*sizeof(int));
    *col = (int*) malloc(nnz_mat*sizeof(int));
    *val = (T*) malloc(nnz_mat*sizeof(T));

    int nnz = 0;

    // Fill local arrays
    for (int i=0; i<ndim; ++i)
    {
        for (int j=0; j<ndim; ++j)
        {
            int idx = i*ndim+j;
            (*rowptr)[idx] = nnz;
            // if no upper boundary element, connect with upper neighbor
            if (i != 0)
            {
                (*col)[nnz] = idx - ndim;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if (j != 0)
            {
                (*col)[nnz] = idx - 1;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // element itself
            (*col)[nnz] = idx;
            (*val)[nnz] = 4.0;
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if (j != ndim - 1)
            {
                (*col)[nnz] = idx + 1;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if (i != ndim - 1)
            {
                (*col)[nnz] = idx + ndim;
                (*val)[nnz] = -1.0;
                ++nnz;
            }
        }
    }
    (*rowptr)[n] = nnz;

    return n;
}

template <typename T>
inline int readMatrixFromMTX(const char *filename,
                             int &nrow, int &ncol, int &nnz,
                             int **row, int **col, T **val)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        return -1;
    }

    char line[1024];

    // Check for banner
    if (!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if (sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for (char *p=array; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=coord; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=data; *p!='\0'; *p=tolower(*p), p++);
    for (char *p=type; *p!='\0'; *p=tolower(*p), p++);

    // Check banner
    if (strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if (strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if (strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if (strcmp(data, "real") != 0)
    {
        return -1;
    }

    // Check type
    if (strcmp(type, "general") != 0 &&
        strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if (line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    int snnz;

    sscanf(line, "%d %d %d", &nrow, &ncol, &snnz);
    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    *row = (int*) malloc(sizeof(int)*nnz);
    *col = (int*) malloc(sizeof(int)*nnz);
    *val = (T*) malloc(sizeof(T)*nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        int irow;
        int icol;
        double dval;

        sscanf(line, "%d %d %lf", &irow, &icol, &dval);

        --irow;
        --icol;

        (*row)[idx] = irow;
        (*col)[idx] = icol;
        (*val)[idx] = (T) dval;

        ++idx;

        if (symm && irow != icol) {

            (*row)[idx] = icol;
            (*col)[idx] = irow;
            (*val)[idx] = (T) dval;

            ++idx;

        }

    }

    fclose(f);

    return 0;
}

template <typename T>
inline void coo_to_csr(int nrow, int ncol, int nnz,
                       const int *src_row, const int *src_col, const T *src_val,
                       int **dst_ptr, int **dst_col, T **dst_val)
{
    *dst_ptr = (int*) malloc(sizeof(int)*(nrow+1));
    *dst_col = (int*) malloc(sizeof(int)*nnz);
    *dst_val = (T*) malloc(sizeof(T)*nnz);

    memset(*dst_ptr, 0, sizeof(int)*(nrow+1));

    // Compute nnz entries per row
    for (int i=0; i<nnz; ++i)
    {
        ++(*dst_ptr)[src_row[i]];
    }

    int sum = 0;
    for (int i=0; i<nrow; ++i)
    {
        int tmp = (*dst_ptr)[i];
        (*dst_ptr)[i] = sum;
        sum += tmp;
    }
    (*dst_ptr)[nrow] = sum;

    // Write column index and values
    for (int i=0; i<nnz; ++i)
    {
        int row = src_row[i];
        int idx = (*dst_ptr)[row];

        (*dst_col)[idx] = src_col[i];
        (*dst_val)[idx] = src_val[i];

        ++(*dst_ptr)[row];
    }

    int last = 0;
    for (int i=0; i<nrow+1; ++i)
    {
        int tmp = (*dst_ptr)[i];
        (*dst_ptr)[i] = last;
        last = tmp;
    }

    for (int i=0; i<nrow; ++i)
    {
        for (int j=(*dst_ptr)[i]; j<(*dst_ptr)[i+1]; ++j)
        {
            for (int k=(*dst_ptr)[i]; k<(*dst_ptr)[i+1]-1; ++k)
            {
                // Swap elements
                int idx = (*dst_col)[k];
                T val = (*dst_val)[k];

                (*dst_col)[k] = (*dst_col)[k+1];
                (*dst_val)[k] = (*dst_val)[k+1];

                (*dst_col)[k+1] = idx;
                (*dst_val)[k+1] = val;
            }
        }
    }
}

#endif // ROCSPARSE_UTILS_H_
