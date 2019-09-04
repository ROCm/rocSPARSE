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

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <algorithm>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocsparse.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

/*!\file
 * \brief provide data initialization and timing utilities.
 */

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#define CHECK_ROCSPARSE_ERROR(error)                              \
    if(error != rocsparse_status_success)                         \
    {                                                             \
        fprintf(stderr, "rocSPARSE error: ");                     \
        if(error == rocsparse_status_invalid_handle)              \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_handle");   \
        }                                                         \
        else if(error == rocsparse_status_not_implemented)        \
        {                                                         \
            fprintf(stderr, " rocsparse_status_not_implemented"); \
        }                                                         \
        else if(error == rocsparse_status_invalid_pointer)        \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_pointer");  \
        }                                                         \
        else if(error == rocsparse_status_invalid_size)           \
        {                                                         \
            fprintf(stderr, "rocsparse_status_invalid_size");     \
        }                                                         \
        else if(error == rocsparse_status_memory_error)           \
        {                                                         \
            fprintf(stderr, "rocsparse_status_memory_error");     \
        }                                                         \
        else if(error == rocsparse_status_internal_error)         \
        {                                                         \
            fprintf(stderr, "rocsparse_status_internal_error");   \
        }                                                         \
        else                                                      \
        {                                                         \
            fprintf(stderr, "rocsparse_status error");            \
        }                                                         \
        fprintf(stderr, "\n");                                    \
        return error;                                             \
    }

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number between [0, 0.999...] . */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return (T)(rand() % 10 + 1); // generate a integer number between [1, 10]
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void rocsparse_init(std::vector<T>& A, rocsparse_int M, rocsparse_int N)
{
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            A[i + j] = random_generator<T>();
        }
    }
};

/* ============================================================================================ */
/*! \brief  vector initialization: */
// initialize sparse index vector with nnz entries ranging from start to end
template <typename I>
void rocsparse_init_index(I* x, rocsparse_int nnz, rocsparse_int start, rocsparse_int end)
{
    std::vector<bool> check(end - start, false);
    rocsparse_int     num = 0;
    while(num < nnz)
    {
        rocsparse_int val = start + rand() % (end - start);
        if(!check[val - start])
        {
            x[num]             = val;
            check[val - start] = true;
            ++num;
        }
    }
    std::sort(x, x + nnz);
};

/* ============================================================================================ */
/*! \brief  csr matrix initialization */
template <typename T>
void rocsparse_init_csr(std::vector<rocsparse_int>& ptr,
                        std::vector<rocsparse_int>& col,
                        std::vector<T>&             val,
                        rocsparse_int               nrow,
                        rocsparse_int               ncol,
                        rocsparse_int               nnz)
{
    // Row offsets
    ptr[0]    = 0;
    ptr[nrow] = nnz;

    for(rocsparse_int i = 1; i < nrow; ++i)
    {
        ptr[i] = rand() % (nnz - 1) + 1;
    }
    std::sort(ptr.begin(), ptr.end());

    // Column indices
    for(rocsparse_int i = 0; i < nrow; ++i)
    {
        rocsparse_init_index(&col[ptr[i]], ptr[i + 1] - ptr[i], 0, ncol - 1);
        std::sort(&col[ptr[i]], &col[ptr[i + 1]]);
    }

    // Random values
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename T>
rocsparse_int gen_2d_laplacian(rocsparse_int               ndim,
                               std::vector<rocsparse_int>& rowptr,
                               std::vector<rocsparse_int>& col,
                               std::vector<T>&             val,
                               rocsparse_index_base        idx_base)
{
    if(ndim == 0)
    {
        return 0;
    }

    rocsparse_int n       = ndim * ndim;
    rocsparse_int nnz_mat = n * 5 - ndim * 4;

    rowptr.resize(n + 1);
    col.resize(nnz_mat);
    val.resize(nnz_mat);

    rocsparse_int nnz = 0;

    // Fill local arrays
    for(rocsparse_int i = 0; i < ndim; ++i)
    {
        for(rocsparse_int j = 0; j < ndim; ++j)
        {
            rocsparse_int idx = i * ndim + j;
            rowptr[idx]       = nnz + idx_base;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                col[nnz] = idx - ndim + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                col[nnz] = idx - 1 + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // element itself
            col[nnz] = idx + idx_base;
            val[nnz] = static_cast<T>(4);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                col[nnz] = idx + 1 + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                col[nnz] = idx + ndim + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz + idx_base;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
void gen_matrix_coo(rocsparse_int               m,
                    rocsparse_int               n,
                    rocsparse_int               nnz,
                    std::vector<rocsparse_int>& row_ind,
                    std::vector<rocsparse_int>& col_ind,
                    std::vector<T>&             val,
                    rocsparse_index_base        idx_base)
{
    if((rocsparse_int)row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if((rocsparse_int)col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if((rocsparse_int)val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Uniform distributed row indices
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        row_ind[i] = rand() % m;
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    rocsparse_int i = 0;
    while(i < nnz)
    {
        rocsparse_int begin = i;
        while(row_ind[i] == row_ind[begin])
        {
            ++i;
            if(i >= nnz)
            {
                break;
            }
        }

        // Sample i disjunct column indices
        rocsparse_int idx = begin;
        while(idx < i)
        {
            // Normal distribution around the diagonal
            rocsparse_int rng = (i - begin) * sqrt(-2.0 * log((double)rand() / RAND_MAX))
                                * cos(2.0 * M_PI * (double)rand() / RAND_MAX);

            if(m <= n)
            {
                rng += row_ind[begin];
            }

            // Repeat if running out of bounds
            if(rng < 0 || rng > n - 1)
            {
                continue;
            }

            // Check for disjunct column index in current row
            if(!check[rng])
            {
                check[rng]   = true;
                col_ind[idx] = rng;
                ++idx;
            }
        }

        // Reset disjunct check array
        for(rocsparse_int j = begin; j < i; ++j)
        {
            check[col_ind[j]] = false;
        }

        // Partially sort column indices
        std::sort(&col_ind[begin], &col_ind[i]);
    }

    // Correct index base accordingly
    if(idx_base == rocsparse_index_base_one)
    {
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    // Sample random values
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>(); //(double) rand() / RAND_MAX;
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <typename T>
rocsparse_int read_mtx_matrix(const char*                 filename,
                              rocsparse_int&              nrow,
                              rocsparse_int&              ncol,
                              rocsparse_int&              nnz,
                              std::vector<rocsparse_int>& row,
                              std::vector<rocsparse_int>& col,
                              std::vector<T>&             val,
                              rocsparse_index_base        idx_base)
{
    printf("Reading matrix %s...", filename);
    fflush(stdout);

    FILE* f = fopen(filename, "r");
    if(!f)
    {
        return -1;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0)
    {
        return -1;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    rocsparse_int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    rocsparse_int snnz;

    int inrow;
    int incol;
    int innz;

    sscanf(line, "%d %d %d", &inrow, &incol, &innz);

    nrow = static_cast<rocsparse_int>(inrow);
    ncol = static_cast<rocsparse_int>(incol);
    snnz = static_cast<rocsparse_int>(innz);

    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    std::vector<rocsparse_int> unsorted_row(nnz);
    std::vector<rocsparse_int> unsorted_col(nnz);
    std::vector<T>             unsorted_val(nnz);

    // Read entries
    rocsparse_int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return true;
        }

        rocsparse_int irow;
        rocsparse_int icol;
        T             ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = static_cast<T>(1);
        }
        else
        {
            ss >> irow >> icol >> ival;
        }

        if(idx_base == rocsparse_index_base_zero)
        {
            --irow;
            --icol;
        }

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                return true;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Sort by row and column index
    std::vector<rocsparse_int> perm(nnz);
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const rocsparse_int& a, const rocsparse_int& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        row[i] = unsorted_row[perm[i]];
        col[i] = unsorted_col[perm[i]];
        val[i] = unsorted_val[perm[i]];
    }

    printf("done.\n");
    fflush(stdout);

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in CSR format */
template <typename T>
rocsparse_int read_bin_matrix(const char*                 filename,
                              rocsparse_int&              nrow,
                              rocsparse_int&              ncol,
                              rocsparse_int&              nnz,
                              std::vector<rocsparse_int>& ptr,
                              std::vector<rocsparse_int>& col,
                              std::vector<T>&             val,
                              rocsparse_index_base        idx_base)
{
    printf("Reading matrix %s...", filename);
    fflush(stdout);

    FILE* f = fopen(filename, "rb");
    if(!f)
    {
        return -1;
    }

    int err;

    int inrow;
    int incol;
    int innz;

    err = fread(&inrow, sizeof(int), 1, f);
    err |= fread(&incol, sizeof(int), 1, f);
    err |= fread(&innz, sizeof(int), 1, f);

    nrow = inrow;
    ncol = incol;
    nnz  = innz;

    // Allocate memory
    ptr.resize(nrow + 1);
    col.resize(nnz);
    val.resize(nnz);

    std::vector<int>    iptr(nrow + 1);
    std::vector<int>    icol(nnz);
    std::vector<double> dval(nnz);

    err |= fread(iptr.data(), sizeof(int), nrow + 1, f);
    err |= fread(icol.data(), sizeof(int), nnz, f);
    err |= fread(dval.data(), sizeof(double), nnz, f);

    fclose(f);

    for(rocsparse_int i = 0; i < nrow + 1; ++i)
    {
        ptr[i] = static_cast<rocsparse_int>(iptr[i]);
    }

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        col[i] = static_cast<rocsparse_int>(icol[i]);
        val[i] = static_cast<T>(dval[i]);
    }

    if(idx_base == rocsparse_index_base_one)
    {
        for(rocsparse_int i = 0; i < nrow + 1; ++i)
        {
            ++ptr[i];
        }

        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++col[i];
        }
    }

    printf("done.\n");
    fflush(stdout);

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in rocALUTION format */
template <typename T>
rocsparse_int read_rocalution_matrix(const char*                 filename,
                                     rocsparse_int&              nrow,
                                     rocsparse_int&              ncol,
                                     rocsparse_int&              nnz,
                                     std::vector<rocsparse_int>& ptr,
                                     std::vector<rocsparse_int>& col,
                                     std::vector<T>&             val,
                                     rocsparse_index_base        idx_base)
{
    printf("Reading matrix %s...", filename);
    fflush(stdout);

    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(!in.is_open())
    {
        return -1;
    }

    std::string header;
    std::getline(in, header);

    if(header != "#rocALUTION binary csr file")
    {
        return -1;
    }

    int version;
    in.read((char*)&version, sizeof(int));

    int inrow;
    int incol;
    int innz;

    in.read((char*)&inrow, sizeof(int));
    in.read((char*)&incol, sizeof(int));
    in.read((char*)&innz, sizeof(int));

    nrow = inrow;
    ncol = incol;
    nnz  = innz;

    // Allocate memory
    ptr.resize(nrow + 1);
    col.resize(nnz);
    val.resize(nnz);

    std::vector<int>    iptr(nrow + 1);
    std::vector<int>    icol(nnz);
    std::vector<double> dval(nnz);

    in.read((char*)iptr.data(), sizeof(int) * (nrow + 1));
    in.read((char*)icol.data(), sizeof(int) * nnz);
    in.read((char*)dval.data(), sizeof(double) * nnz);

    in.close();

    for(rocsparse_int i = 0; i < nrow + 1; ++i)
    {
        ptr[i] = static_cast<rocsparse_int>(iptr[i]);
    }

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        col[i] = static_cast<rocsparse_int>(icol[i]);
        val[i] = static_cast<T>(dval[i]);
    }

    if(idx_base == rocsparse_index_base_one)
    {
        for(rocsparse_int i = 0; i < nrow + 1; ++i)
        {
            ++ptr[i];
        }

        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++col[i];
        }
    }

    printf("done.\n");
    fflush(stdout);

    return 0;
}

/* ============================================================================================ */
/*! \brief  Compute incomplete LU factorization without fill-ins and no pivoting using CSR
 *  matrix storage format.
 */
template <typename T>
rocsparse_int csrilu0(rocsparse_int        m,
                      const rocsparse_int* ptr,
                      const rocsparse_int* col,
                      T*                   val,
                      rocsparse_index_base idx_base)
{
    // pointer of upper part of each row
    std::vector<rocsparse_int> diag_offset(m);
    std::vector<rocsparse_int> nnz_entries(m, 0);

    // ai = 0 to N loop over all rows
    for(rocsparse_int ai = 0; ai < m; ++ai)
    {
        // ai-th row entries
        rocsparse_int row_start = ptr[ai] - idx_base;
        rocsparse_int row_end   = ptr[ai + 1] - idx_base;
        rocsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(col[j] - idx_base < ai)
            {

                rocsparse_int col_j  = col[j] - idx_base;
                rocsparse_int diag_j = diag_offset[col_j];

                if(val[diag_j] != static_cast<T>(0))
                {
                    // multiplication factor
                    val[j] = val[j] / val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(rocsparse_int k = diag_j + 1; k < ptr[col_j + 1] - idx_base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[col[k] - idx_base] != 0)
                        {
                            rocsparse_int idx = nnz_entries[col[k] - idx_base];
                            val[idx]          = std::fma(-val[j], val[k], val[idx]);
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    return col_j + idx_base;
                }
            }
            else if(col[j] - idx_base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural zero digonal
            return ai + idx_base;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = 0;
        }
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular lower solve using CSR storage format. */
template <typename T>
rocsparse_int lsolve(rocsparse_int        m,
                     const rocsparse_int* ptr,
                     const rocsparse_int* col,
                     const T*             val,
                     T                    alpha,
                     const T*             x,
                     T*                   y,
                     rocsparse_index_base idx_base,
                     rocsparse_diag_type  diag_type,
                     unsigned int         wf_size)
{
    rocsparse_int  pivot = std::numeric_limits<rocsparse_int>::max();
    std::vector<T> temp(wf_size);

    for(rocsparse_int i = 0; i < m; ++i)
    {
        temp.assign(wf_size, static_cast<T>(0));
        temp[0] = alpha * x[i];

        rocsparse_int diag      = -1;
        rocsparse_int row_begin = ptr[i] - idx_base;
        rocsparse_int row_end   = ptr[i + 1] - idx_base;

        T diag_val = static_cast<T>(0);

        for(rocsparse_int l = row_begin; l < row_end; l += wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                rocsparse_int j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                rocsparse_int col_j = col[j] - idx_base;
                T             val_j = val[j];

                if(col_j < i)
                {
                    // Lower part
                    temp[k] = std::fma(-val[j], y[col_j], temp[k]);
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(val_j == static_cast<T>(0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / val_j;
                    }

                    break;
                }
                else
                {
                    // Upper part
                    break;
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = temp[0] * diag_val;
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != std::numeric_limits<rocsparse_int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular upper solve using CSR storage format. */
template <typename T>
rocsparse_int usolve(rocsparse_int        m,
                     const rocsparse_int* ptr,
                     const rocsparse_int* col,
                     const T*             val,
                     T                    alpha,
                     const T*             x,
                     T*                   y,
                     rocsparse_index_base idx_base,
                     rocsparse_diag_type  diag_type,
                     unsigned int         wf_size)
{
    rocsparse_int  pivot = std::numeric_limits<rocsparse_int>::max();
    std::vector<T> temp(wf_size);

    for(rocsparse_int i = m - 1; i >= 0; --i)
    {
        temp.assign(wf_size, static_cast<T>(0));
        temp[0] = alpha * x[i];

        rocsparse_int diag      = -1;
        rocsparse_int row_begin = ptr[i] - idx_base;
        rocsparse_int row_end   = ptr[i + 1] - idx_base;

        T diag_val = static_cast<T>(0);

        for(rocsparse_int l = row_end - 1; l >= row_begin; l -= wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                rocsparse_int j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                rocsparse_int col_j = col[j] - idx_base;
                T             val_j = val[j];

                if(col_j < i)
                {
                    // Lower part
                    continue;
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(val_j == static_cast<T>(0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / val_j;
                    }

                    continue;
                }
                else
                {
                    // Upper part
                    temp[k] = std::fma(-val[j], y[col_j], temp[k]);
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = temp[0] * diag_val;
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != std::numeric_limits<rocsparse_int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Transpose sparse matrix using CSR storage format. */
template <typename T>
void transpose(rocsparse_int        m,
               rocsparse_int        n,
               rocsparse_int        nnz,
               const rocsparse_int* csr_row_ptr_A,
               const rocsparse_int* csr_col_ind_A,
               const T*             csr_val_A,
               rocsparse_int*       csr_row_ptr_B,
               rocsparse_int*       csr_col_ind_B,
               T*                   csr_val_B,
               rocsparse_index_base idx_base_A,
               rocsparse_index_base idx_base_B)
{
    memset(csr_row_ptr_B, 0, sizeof(rocsparse_int) * (n + 1));

    // Determine nnz per column
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr_B[csr_col_ind_A[i] + 1 - idx_base_A];
    }

    // Scan
    for(rocsparse_int i = 0; i < n; ++i)
    {
        csr_row_ptr_B[i + 1] += csr_row_ptr_B[i];
    }

    // Fill row indices and values
    for(rocsparse_int i = 0; i < m; ++i)
    {
        rocsparse_int row_begin = csr_row_ptr_A[i] - idx_base_A;
        rocsparse_int row_end   = csr_row_ptr_A[i + 1] - idx_base_A;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int col = csr_col_ind_A[j] - idx_base_A;
            rocsparse_int idx = csr_row_ptr_B[col];

            csr_col_ind_B[idx] = i + idx_base_B;
            csr_val_B[idx]     = csr_val_A[j];

            ++csr_row_ptr_B[col];
        }
    }

    // Shift column pointer array
    for(rocsparse_int i = n; i > 0; --i)
    {
        csr_row_ptr_B[i] = csr_row_ptr_B[i - 1] + idx_base_B;
    }

    csr_row_ptr_B[0] = idx_base_B;
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for rocsparse version and git commit SHA-1. */
void query_version(char* version);

/* ============================================================================================ */
/*  device query and print out their ID and name */
rocsparse_int query_device_property();

/*  set current device to device_id */
void set_device(rocsparse_int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this rocsparse library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
public:
    rocsparse_int M   = 128;
    rocsparse_int N   = 128;
    rocsparse_int K   = 128;
    rocsparse_int nnz = 32;

    rocsparse_int ldb;
    rocsparse_int ldc;

    double alpha = 1.0;
    double beta  = 0.0;

    rocsparse_operation       transA    = rocsparse_operation_none;
    rocsparse_operation       transB    = rocsparse_operation_none;
    rocsparse_index_base      idx_base  = rocsparse_index_base_zero;
    rocsparse_index_base      idx_base2 = rocsparse_index_base_zero;
    rocsparse_index_base      idx_base3 = rocsparse_index_base_zero;
    rocsparse_index_base      idx_base4 = rocsparse_index_base_zero;
    rocsparse_action          action    = rocsparse_action_numeric;
    rocsparse_hyb_partition   part      = rocsparse_hyb_partition_auto;
    rocsparse_diag_type       diag_type = rocsparse_diag_type_non_unit;
    rocsparse_fill_mode       fill_mode = rocsparse_fill_mode_lower;
    rocsparse_analysis_policy analysis  = rocsparse_analysis_policy_reuse;

    rocsparse_int norm_check = 0;
    rocsparse_int unit_check = 1;
    rocsparse_int timing     = 0;
    rocsparse_int laplacian  = 0;

    int iters     = 10;
    int ell_width = 0;
    int temp      = 0;

    std::string filename   = "";
    std::string rocalution = "";
    bool        bswitch    = false;

    Arguments& operator=(const Arguments& rhs)
    {
        this->M   = rhs.M;
        this->N   = rhs.N;
        this->K   = rhs.K;
        this->nnz = rhs.nnz;

        this->ldb = rhs.ldb;
        this->ldc = rhs.ldc;

        this->alpha = rhs.alpha;
        this->beta  = rhs.beta;

        this->transA    = rhs.transA;
        this->transB    = rhs.transB;
        this->idx_base  = rhs.idx_base;
        this->idx_base2 = rhs.idx_base2;
        this->idx_base3 = rhs.idx_base3;
        this->idx_base4 = rhs.idx_base4;
        this->action    = rhs.action;
        this->part      = rhs.part;
        this->diag_type = rhs.diag_type;
        this->fill_mode = rhs.fill_mode;
        this->analysis  = rhs.analysis;

        this->norm_check = rhs.norm_check;
        this->unit_check = rhs.unit_check;
        this->timing     = rhs.timing;

        this->iters     = rhs.iters;
        this->laplacian = rhs.laplacian;
        this->ell_width = rhs.ell_width;
        this->temp      = rhs.temp;

        this->filename   = rhs.filename;
        this->rocalution = rhs.rocalution;
        this->bswitch    = rhs.bswitch;

        return *this;
    }
};

#endif // TESTING_UTILITY_HPP
