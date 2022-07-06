/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_matrix_factory_file.hpp"
#include "rocsparse_import.hpp"
#include "rocsparse_importer_impls.hpp"
#include "rocsparse_matrix_utils.hpp"

template <typename T, template <typename...> class VECTOR>
static void apply_toint(VECTOR<T>& data)
{
    const size_t size = data.size();
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = std::abs(data[i]);
    }
}

template <template <typename...> class VECTOR>
static void apply_toint(VECTOR<rocsparse_float_complex>& data)
{
    const size_t size = data.size();
    for(size_t i = 0; i < size; ++i)
    {
        rocsparse_float_complex c = data[i];
        data[i] = rocsparse_float_complex(std::abs(static_cast<float>(std::real(c))),
                                          std::abs(static_cast<float>(std::imag(c))));
    }
}

template <template <typename...> class VECTOR>
static void apply_toint(VECTOR<rocsparse_double_complex>& data)
{
    const size_t size = data.size();
    for(size_t i = 0; i < size; ++i)
    {
        rocsparse_double_complex c = data[i];
        data[i] = rocsparse_double_complex(std::abs(static_cast<double>(std::real(c))),
                                           std::abs(static_cast<double>(std::imag(c))));
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
template <rocsparse_matrix_init MATRIX_INIT>
struct rocsparse_init_file_traits;

template <>
struct rocsparse_init_file_traits<rocsparse_matrix_file_rocalution>
{
    using importer_t = rocsparse_importer_rocalution;
};

template <>
struct rocsparse_init_file_traits<rocsparse_matrix_file_mtx>
{
    using importer_t = rocsparse_importer_matrixmarket;
};

template <>
struct rocsparse_init_file_traits<rocsparse_matrix_file_rocsparseio>
{
    using importer_t = rocsparse_importer_rocsparseio;
};

template <rocsparse_matrix_init MATRIX_INIT>
struct rocsparse_init_file
{
    using importer_t = typename rocsparse_init_file_traits<MATRIX_INIT>::importer_t;

    template <typename... S>
    static inline rocsparse_status import_gebsr(const char* filename, S&&... s)
    {
        importer_t importer(filename);
        return rocsparse_import_sparse_gebsr(importer, s...);
    }

    template <typename... S>
    static inline rocsparse_status import_csr(const char* filename, S&&... s)
    {

        importer_t importer(filename);
        return rocsparse_import_sparse_csr(importer, s...);
    }

    template <typename... S>
    static inline rocsparse_status import_coo(const char* filename, S&&... s)
    {
        importer_t importer(filename);
        return rocsparse_import_sparse_coo(importer, s...);
    }
};

template <rocsparse_matrix_init MATRIX_INIT, typename T, typename I, typename J>
rocsparse_matrix_factory_file<MATRIX_INIT, T, I, J>::rocsparse_matrix_factory_file(
    const char* filename, bool toint)
    : m_filename(filename)
    , m_toint(toint){};

template <typename T, typename I, typename J>
struct spec
{
    template <rocsparse_matrix_init MATRIX_INIT>
    static void init_gebsr_rocalution(rocsparse_matrix_factory_file<MATRIX_INIT, T, I, J>& factory,
                                      std::vector<I>&        bsr_row_ptr,
                                      std::vector<J>&        bsr_col_ind,
                                      std::vector<T>&        bsr_val,
                                      rocsparse_direction    dirb,
                                      J&                     Mb,
                                      J&                     Nb,
                                      I&                     nnzb,
                                      J&                     row_block_dim,
                                      J&                     col_block_dim,
                                      rocsparse_index_base   base,
                                      rocsparse_matrix_type  matrix_type,
                                      rocsparse_fill_mode    uplo,
                                      rocsparse_storage_mode storage)
    {
        factory.init_csr(
            bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base, matrix_type, uplo, storage);

        // Then temporarily skip the values.
        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }
};

template <typename T>
struct spec<T, rocsparse_int, rocsparse_int>
{
    template <rocsparse_matrix_init MATRIX_INIT>
    static void init_gebsr_rocalution(
        rocsparse_matrix_factory_file<MATRIX_INIT, T, rocsparse_int, rocsparse_int>& factory,
        std::vector<rocsparse_int>&                                                  bsr_row_ptr,
        std::vector<rocsparse_int>&                                                  bsr_col_ind,
        std::vector<T>&                                                              bsr_val,
        rocsparse_direction                                                          dirb,
        rocsparse_int&                                                               Mb,
        rocsparse_int&                                                               Nb,
        rocsparse_int&                                                               nnzb,
        rocsparse_int&                                                               row_block_dim,
        rocsparse_int&                                                               col_block_dim,
        rocsparse_index_base                                                         base,
        rocsparse_matrix_type                                                        matrix_type,
        rocsparse_fill_mode                                                          uplo,
        rocsparse_storage_mode                                                       storage)
    {
        //
        // Initialize in case init_csr requires it as input.
        //
        rocsparse_int M = Mb * row_block_dim;
        rocsparse_int N = Nb * col_block_dim;

        host_csr_matrix<T, rocsparse_int, rocsparse_int> hA_uncompressed(M, N, 0, base);
        factory.init_csr(hA_uncompressed.ptr,
                         hA_uncompressed.ind,
                         hA_uncompressed.val,
                         hA_uncompressed.m,
                         hA_uncompressed.n,
                         hA_uncompressed.nnz,
                         hA_uncompressed.base,
                         matrix_type,
                         uplo,
                         storage);

        device_gebsr_matrix<T, rocsparse_int, rocsparse_int> that_on_device;
        {
            device_csr_matrix<T, rocsparse_int, rocsparse_int> dA_uncompressed(hA_uncompressed);
            rocsparse_matrix_utils::convert(
                dA_uncompressed, dirb, row_block_dim, col_block_dim, base, that_on_device);
        }

        Mb            = that_on_device.mb;
        Nb            = that_on_device.nb;
        nnzb          = that_on_device.nnzb;
        row_block_dim = that_on_device.row_block_dim;
        col_block_dim = that_on_device.col_block_dim;
        that_on_device.ptr.transfer_to(bsr_row_ptr);
        that_on_device.ind.transfer_to(bsr_col_ind);
        that_on_device.val.transfer_to(bsr_val);
    }
};

template <rocsparse_matrix_init MATRIX_INIT, typename T, typename I, typename J>
void rocsparse_matrix_factory_file<MATRIX_INIT, T, I, J>::init_gebsr(
    std::vector<I>&        bsr_row_ptr,
    std::vector<J>&        bsr_col_ind,
    std::vector<T>&        bsr_val,
    rocsparse_direction    dirb,
    J&                     Mb,
    J&                     Nb,
    I&                     nnzb,
    J&                     row_block_dim,
    J&                     col_block_dim,
    rocsparse_index_base   base,
    rocsparse_matrix_type  matrix_type,
    rocsparse_fill_mode    uplo,
    rocsparse_storage_mode storage)
{
    switch(MATRIX_INIT)
    {
    case rocsparse_matrix_file_mtx:
    {
        this->init_csr(
            bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base, matrix_type, uplo, storage);
        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
        break;
    }
    case rocsparse_matrix_file_rocalution:
    {
        spec<T, I, J>::init_gebsr_rocalution(*this,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_val,
                                             dirb,
                                             Mb,
                                             Nb,
                                             nnzb,
                                             row_block_dim,
                                             col_block_dim,
                                             base,
                                             matrix_type,
                                             uplo,
                                             storage);
        break;
    }
    case rocsparse_matrix_file_rocsparseio:
    {
        rocsparse_direction import_dirb = {};
        rocsparse_status    status
            = rocsparse_init_file<MATRIX_INIT>::import_gebsr(this->m_filename.c_str(),
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             bsr_val,
                                                             import_dirb,
                                                             Mb,
                                                             Nb,
                                                             nnzb,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             base);
        CHECK_ROCSPARSE_THROW_ERROR(status);

        if(import_dirb != dirb)
        {
            std::cerr << "TODO, reorder ?" << std::endl;
            exit(1);
        }
        break;
    }
    }

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_gebsrunsort<T, I, J>(
            bsr_row_ptr.data(), bsr_col_ind.data(), Mb, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }

    if(this->m_toint)
    {
        apply_toint(bsr_val);
    }
}

template <rocsparse_matrix_init MATRIX_INIT, typename T, typename I, typename J>
void rocsparse_matrix_factory_file<MATRIX_INIT, T, I, J>::init_csr(
    std::vector<I>&        csr_row_ptr,
    std::vector<J>&        csr_col_ind,
    std::vector<T>&        csr_val,
    J&                     M,
    J&                     N,
    I&                     nnz,
    rocsparse_index_base   base,
    rocsparse_matrix_type  matrix_type,
    rocsparse_fill_mode    uplo,
    rocsparse_storage_mode storage)
{
    std::vector<I> row_ptr;
    std::vector<J> col_ind;
    std::vector<T> val;

    switch(MATRIX_INIT)
    {
    case rocsparse_matrix_file_rocalution:
    {
        rocsparse_status status = rocsparse_init_file<rocsparse_matrix_file_rocalution>::import_csr(
            this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        CHECK_ROCSPARSE_THROW_ERROR(status);
        break;
    }

    case rocsparse_matrix_file_rocsparseio:
    {
        rocsparse_status status
            = rocsparse_init_file<rocsparse_matrix_file_rocsparseio>::import_csr(
                this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        CHECK_ROCSPARSE_THROW_ERROR(status);
        break;
    }
    case rocsparse_matrix_file_mtx:
    {
        I              coo_M, coo_N;
        std::vector<I> coo_row_ind;
        std::vector<I> coo_col_ind;

        // Read COO matrix
        this->init_coo(
            coo_row_ind, coo_col_ind, val, coo_M, coo_N, nnz, base, matrix_type, uplo, storage);

        // Convert to CSR
        M = (J)coo_M;
        N = (J)coo_N;

        row_ptr.resize(M + 1);
        col_ind.resize(nnz);

        host_coo_to_csr(coo_M, nnz, coo_row_ind.data(), row_ptr, base);
        for(I i = 0; i < nnz; ++i)
        {
            col_ind[i] = (J)coo_col_ind[i];
        }

        break;
    }
    }

    switch(matrix_type)
    {
    case rocsparse_matrix_type_general:
    {
        csr_row_ptr = row_ptr;
        csr_col_ind = col_ind;
        csr_val     = val;
        break;
    }
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        rocsparse_matrix_utils::host_csrtri(row_ptr.data(),
                                            col_ind.data(),
                                            val.data(),
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            M,
                                            N,
                                            nnz,
                                            base,
                                            uplo);
        break;
    }
    }

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_csrunsort<T, I, J>(
            csr_row_ptr.data(), csr_col_ind.data(), M, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }

    //
    // Apply toint?
    //
    if(this->m_toint)
    {
        apply_toint(csr_val);
    }
}

template <rocsparse_matrix_init MATRIX_INIT, typename T, typename I, typename J>
void rocsparse_matrix_factory_file<MATRIX_INIT, T, I, J>::init_coo(
    std::vector<I>&        coo_row_ind,
    std::vector<I>&        coo_col_ind,
    std::vector<T>&        coo_val,
    I&                     M,
    I&                     N,
    I&                     nnz,
    rocsparse_index_base   base,
    rocsparse_matrix_type  matrix_type,
    rocsparse_fill_mode    uplo,
    rocsparse_storage_mode storage)
{
    std::vector<I> row_ind;
    std::vector<I> col_ind;
    std::vector<T> val;

    switch(MATRIX_INIT)
    {
    case rocsparse_matrix_file_rocalution:
    {
        std::vector<I>   row_ptr(M + 1);
        rocsparse_status status = rocsparse_init_file<rocsparse_matrix_file_rocalution>::import_csr(
            this->m_filename.c_str(), row_ptr, col_ind, val, M, N, nnz, base);
        CHECK_ROCSPARSE_THROW_ERROR(status);

        //
        // Convert to COO
        //
        host_csr_to_coo(M, nnz, row_ptr, row_ind, base);
        break;
    }

    case rocsparse_matrix_file_mtx:
    {
        rocsparse_status status = rocsparse_init_file<rocsparse_matrix_file_mtx>::import_coo(
            this->m_filename.c_str(), row_ind, col_ind, val, M, N, nnz, base);
        CHECK_ROCSPARSE_THROW_ERROR(status);
        break;
    }

    case rocsparse_matrix_file_rocsparseio:
    {
        rocsparse_status status
            = rocsparse_init_file<rocsparse_matrix_file_rocsparseio>::import_coo(
                this->m_filename.c_str(), row_ind, col_ind, val, M, N, nnz, base);
        CHECK_ROCSPARSE_THROW_ERROR(status);
        break;
    }
    }

    switch(matrix_type)
    {
    case rocsparse_matrix_type_general:
    {
        coo_row_ind = row_ind;
        coo_col_ind = col_ind;
        coo_val     = val;
        break;
    }
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        rocsparse_matrix_utils::host_cootri(row_ind.data(),
                                            col_ind.data(),
                                            val.data(),
                                            coo_row_ind,
                                            coo_col_ind,
                                            coo_val,
                                            M,
                                            N,
                                            nnz,
                                            base,
                                            uplo);
        break;
    }
    }

    switch(storage)
    {
    case rocsparse_storage_mode_unsorted:
    {
        rocsparse_matrix_utils::host_coounsort<T, I>(
            coo_row_ind.data(), coo_col_ind.data(), M, nnz, base);
        break;
    }
    case rocsparse_storage_mode_sorted:
    {
        break;
    }
    }

    if(this->m_toint)
    {
        apply_toint(coo_val);
    }
}

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, float, int32_t, int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, float, int64_t, int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, float, int64_t, int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, double, int32_t, int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, double, int64_t, int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx, double, int64_t, int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_float_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_double_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_mtx,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              float,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              float,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              float,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              double,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              double,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              double,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_float_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_double_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocalution,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              float,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              float,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              float,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              double,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              double,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              double,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_float_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_float_complex,
                                              int64_t,
                                              int64_t>;

template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_double_complex,
                                              int32_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int32_t>;
template struct rocsparse_matrix_factory_file<rocsparse_matrix_file_rocsparseio,
                                              rocsparse_double_complex,
                                              int64_t,
                                              int64_t>;
