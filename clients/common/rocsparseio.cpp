#include "rocsparseio.hpp"
#include <iostream>

extern "C" rocsparseio_status rocsparseio_open(rocsparseio_handle* p_handle,
                                               rocsparseio_rwmode  mode,
                                               const char*         filename,
                                               ...)
{
    ROCSPARSEIO_C_CHECK_ARG(!p_handle, rocsparseio_status_invalid_handle);

    {
        rocsparseio::rwmode_t rwmode = mode;
        ROCSPARSEIO_C_CHECK_ARG(rwmode.is_invalid(), rocsparseio_status_invalid_value);
    }

    ROCSPARSEIO_C_CHECK_ARG(filename == nullptr, rocsparseio_status_invalid_pointer);

    {
        va_list args;
        va_start(args, filename);
        ROCSPARSEIO_C_CHECK(rocsparseio::open(p_handle, mode, filename, args));
        va_end(args);
    }

    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_close(rocsparseio_handle handle)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio_status_invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::close(handle));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_format(rocsparseio_handle  handle,
                                                      rocsparseio_format* format)
{
    rocsparseio::format_t f;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_format(handle, &f));
    format[0] = (rocsparseio_format)f;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_type_get_size(rocsparseio_type type_, uint64_t* p_size_)
{
    ROCSPARSEIO_C_CHECK_ARG(!p_size_, rocsparseio_status_invalid_pointer);
    rocsparseio::type_t type = type_;
    p_size_[0]               = type.size();
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_dense_vector(rocsparseio_handle handle,
                                                             rocsparseio_type   data_type,
                                                             uint64_t           data_nmemb,
                                                             const void*        data,
                                                             uint64_t           data_ld)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(data_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);

    ROCSPARSEIO_C_CHECK_ARG((0 != data_nmemb) && (nullptr == data),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(
        rocsparseio::write_dense_vector(handle, data_type, data_nmemb, data, data_ld));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_dense_vector(rocsparseio_handle handle,
                                                                      rocsparseio_type*  data_type,
                                                                      uint64_t*          data_nmemb)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_nmemb, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_vector(handle, &type, data_nmemb));
    data_type[0] = (rocsparseio_type)type;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_dense_vector(rocsparseio_handle handle, void* data, uint64_t data_inc)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(!data, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_dense_vector(handle, data, data_inc));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_dense_vector(rocsparseio_handle handle,
                                                            rocsparseio_type*  data_type,
                                                            uint64_t*          data_nmemb,
                                                            void**             data)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_nmemb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_dense_vector(handle, data_type, data_nmemb));
    data[0] = nullptr;
    if(data_nmemb[0] > 0)
    {
        data[0] = malloc(rocsparseio::type_t(data_type[0]).size() * data_nmemb[0]);
        // LCOV_EXCL_START
        if(!data[0])
        {
            return rocsparseio_status_invalid_memory;
        }
        // LCOV_EXCL_STOP
        ROCSPARSEIO_C_CHECK(rocsparseiox_read_dense_vector(handle, data[0], 1));
    }
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_dense_matrix(rocsparseio_handle handle,
                                                            rocsparseio_order* data_order,
                                                            uint64_t*          m,
                                                            uint64_t*          n,
                                                            rocsparseio_type*  data_type,
                                                            void**             data)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_order, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data, rocsparseio::status_t::invalid_pointer);

    rocsparseio::order_t order;
    rocsparseio::type_t  type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_matrix(handle, &order, m, n, &type));

    data_order[0] = (rocsparseio_order)order;
    data_type[0]  = (rocsparseio_type)type;
    data[0]       = malloc(type.size() * m[0] * n[0]);
    // LCOV_EXCL_START
    if(!data[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_dense_matrix(
        handle, data[0], (order == rocsparseio::order_t::row) ? n[0] : m[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_dense_matrix(rocsparseio_handle handle,
                                                                      rocsparseio_order* data_order,
                                                                      uint64_t*          m,
                                                                      uint64_t*          n,
                                                                      rocsparseio_type*  data_type)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_order, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type, rocsparseio::status_t::invalid_pointer);
    rocsparseio::order_t order;
    rocsparseio::type_t  type;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_dense_matrix(handle, &order, m, n, &type));
    data_order[0] = (rocsparseio_order)order;
    data_type[0]  = (rocsparseio_type)type;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_dense_matrix(rocsparseio_handle handle, void* data, uint64_t data_ld)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_dense_matrix(handle, data));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_dense_matrix(rocsparseio_handle handle,
                                                             rocsparseio_order  order,
                                                             uint64_t           m,
                                                             uint64_t           n,
                                                             rocsparseio_type   data_type,
                                                             const void*        data,
                                                             uint64_t           data_ld)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::order_t(order).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(data_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((0 != m) && (0 != n) && (nullptr == data),
                            rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((order == rocsparseio_order_row) && (data_ld < n),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((order == rocsparseio_order_column) && (data_ld < m),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK(
        rocsparseio::write_dense_matrix(handle, order, m, n, data_type, data, data_ld));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_coo(rocsparseio_handle handle,
                                                           uint64_t           m,
                                                           uint64_t           n,
                                                           uint64_t           nnz,

                                                           rocsparseio_type row_ind_type,
                                                           const void*      row_ind,

                                                           rocsparseio_type col_ind_type,
                                                           const void*      col_ind,

                                                           rocsparseio_type       val_type,
                                                           const void*            val,
                                                           rocsparseio_index_base base)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(row_ind_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(col_ind_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((nnz > 0 && !row_ind), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz > 0 && !col_ind), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz > 0 && !val), rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_coo(
        handle, m, n, nnz, row_ind_type, row_ind, col_ind_type, col_ind, val_type, val, base));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_coo(rocsparseio_handle handle,
                                                                    uint64_t*          m,
                                                                    uint64_t*          n,
                                                                    uint64_t*          nnz,

                                                                    rocsparseio_type* row_ind_type_,
                                                                    rocsparseio_type* col_ind_type_,
                                                                    rocsparseio_type* data_type_,
                                                                    rocsparseio_index_base* base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnz, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::type_t       row_ind_type;
    rocsparseio::type_t       col_ind_type;
    rocsparseio::type_t       data_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_coo(
        handle, m, n, nnz, &row_ind_type, &col_ind_type, &data_type, &index_base));
    row_ind_type_[0] = (rocsparseio_type)row_ind_type;
    col_ind_type_[0] = (rocsparseio_type)col_ind_type;
    data_type_[0]    = (rocsparseio_type)data_type;
    base_[0]         = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_sparse_coo(rocsparseio_handle handle,
                                                           void*              row_ind,
                                                           void*              col_ind,
                                                           void*              data)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_coo(handle, row_ind, col_ind, data));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_coo(rocsparseio_handle handle,
                                                          uint64_t*          m,
                                                          uint64_t*          n,
                                                          uint64_t*          nnz,

                                                          rocsparseio_type* row_ind_type_,
                                                          void**            row_ind,

                                                          rocsparseio_type* col_ind_type_,
                                                          void**            col_ind,

                                                          rocsparseio_type*       val_type_,
                                                          void**                  val,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnz, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base_, rocsparseio::status_t::invalid_pointer);

    rocsparseio::type_t       row_ind_type;
    rocsparseio::type_t       col_ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_coo(
        handle, m, n, nnz, &row_ind_type, &col_ind_type, &val_type, &index_base));
    row_ind_type_[0] = (rocsparseio_type)row_ind_type;
    col_ind_type_[0] = (rocsparseio_type)col_ind_type;
    val_type_[0]     = (rocsparseio_type)val_type;
    base_[0]         = (rocsparseio_index_base)index_base;

    row_ind[0] = malloc(row_ind_type.size() * nnz[0]);
    // LCOV_EXCL_START
    if(!row_ind[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    col_ind[0] = malloc(col_ind_type.size() * (nnz[0]));
    // LCOV_EXCL_START
    if(!col_ind[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    val[0] = malloc(val_type.size() * (nnz[0]));
    // LCOV_EXCL_START
    if(!val[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_coo(handle, row_ind[0], col_ind[0], val[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_csx(rocsparseio_handle    handle,
                                                           rocsparseio_direction dir_,
                                                           uint64_t              m,
                                                           uint64_t              n,
                                                           uint64_t              nnz,

                                                           rocsparseio_type ptr_type,
                                                           const void*      ptr,

                                                           rocsparseio_type ind_type,
                                                           const void*      ind,

                                                           rocsparseio_type       val_type,
                                                           const void*            val,
                                                           rocsparseio_index_base base)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dir_).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ptr_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((m > 0 && !ptr), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz > 0 && !ind), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnz > 0 && !val), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_csx(
        handle, dir_, m, n, nnz, ptr_type, ptr, ind_type, ind, val_type, val, base));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_csx(rocsparseio_handle     handle,
                                                                    rocsparseio_direction* dir_,
                                                                    uint64_t*              m,
                                                                    uint64_t*              n,
                                                                    uint64_t*              nnz,

                                                                    rocsparseio_type* ptr_type_,
                                                                    rocsparseio_type* ind_type_,
                                                                    rocsparseio_type* data_type_,
                                                                    rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnz, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::direction_t  dir;
    rocsparseio::type_t       ptr_type;
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       data_type;
    rocsparseio::index_base_t index_base;

    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_csx(
        handle, &dir, m, n, nnz, &ptr_type, &ind_type, &data_type, &index_base));
    dir_[0]       = (rocsparseio_direction)dir;
    ptr_type_[0]  = (rocsparseio_type)ptr_type;
    ind_type_[0]  = (rocsparseio_type)ind_type;
    data_type_[0] = (rocsparseio_type)data_type;
    base_[0]      = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_csx(rocsparseio_handle handle, void* ptr, void* ind, void* data)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == data, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_csx(handle, ptr, ind, data));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_csx(rocsparseio_handle     handle,
                                                          rocsparseio_direction* dir_,
                                                          uint64_t*              m,
                                                          uint64_t*              n,
                                                          uint64_t*              nnz,

                                                          rocsparseio_type* ptr_type_,
                                                          void**            ptr,

                                                          rocsparseio_type* ind_type_,
                                                          void**            ind,

                                                          rocsparseio_type*       val_type_,
                                                          void**                  val,
                                                          rocsparseio_index_base* base_)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == m, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == n, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnz, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base_, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_sparse_csx(
        handle, dir_, m, n, nnz, ptr_type_, ind_type_, val_type_, base_));
    uint64_t                 ptr_size  = 0;
    rocsparseio::type_t      ind_type  = ind_type_[0];
    rocsparseio::type_t      data_type = val_type_[0];
    rocsparseio::type_t      ptr_type  = ptr_type_[0];
    rocsparseio::direction_t dir       = dir_[0];
    switch(dir)
    {
    case rocsparseio::direction_t::row:
    {
        ptr_size = m[0] + 1;
        break;
    }
    case rocsparseio::direction_t::column:
    {
        ptr_size = n[0] + 1;
        break;
    }
    }

    ptr[0] = malloc(ptr_type.size() * ptr_size);
    // LCOV_EXCL_START
    if(!ptr[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    ind[0] = malloc(ind_type.size() * (nnz[0]));
    // LCOV_EXCL_START
    if(!ind[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    val[0] = malloc(data_type.size() * (nnz[0]));
    // LCOV_EXCL_START
    if(!val[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_sparse_csx(handle, ptr[0], ind[0], val[0]));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_write_sparse_gebsx(rocsparseio_handle handle,

                                                             rocsparseio_direction dir,
                                                             rocsparseio_direction dirb,

                                                             uint64_t mb,
                                                             uint64_t nb,
                                                             uint64_t nnzb,

                                                             uint64_t row_block_dim,
                                                             uint64_t col_block_dim,

                                                             rocsparseio_type ptr_type,
                                                             const void*      ptr,

                                                             rocsparseio_type ind_type,
                                                             const void*      ind,

                                                             rocsparseio_type       val_type,
                                                             const void*            val,
                                                             rocsparseio_index_base base)
{
    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dir).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::direction_t(dirb).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ptr_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(ind_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::type_t(val_type).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG(rocsparseio::index_base_t(base).is_invalid(),
                            rocsparseio::status_t::invalid_value);
    ROCSPARSEIO_C_CHECK_ARG((mb > 0 && !ptr), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnzb > 0 && !ind), rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG((nnzb > 0 && !val), rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK_ARG(0 == handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK(rocsparseio::write_sparse_gebsx(handle,
                                                        dir,
                                                        dirb,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        row_block_dim,
                                                        col_block_dim,
                                                        ptr_type,
                                                        ptr,
                                                        ind_type,
                                                        ind,
                                                        val_type,
                                                        val,
                                                        base));
    return rocsparseio_status_success;
}
extern "C" rocsparseio_status rocsparseiox_read_metadata_sparse_gebsx(rocsparseio_handle     handle,
                                                                      rocsparseio_direction* dir_,
                                                                      rocsparseio_direction* dirb_,
                                                                      uint64_t*              mb,
                                                                      uint64_t*              nb,
                                                                      uint64_t*              nnzb,
                                                                      uint64_t* row_block_dim,
                                                                      uint64_t* col_block_dim,
                                                                      rocsparseio_type* ptr_type_,
                                                                      rocsparseio_type* ind_type_,
                                                                      rocsparseio_type* val_type_,
                                                                      rocsparseio_index_base* base_)
{

    ROCSPARSEIO_C_CHECK_ARG(!handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dir_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dirb_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == mb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnzb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_block_dim, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_block_dim, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val_type_, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base_, rocsparseio::status_t::invalid_pointer);
    rocsparseio::direction_t  dir;
    rocsparseio::direction_t  dirb;
    rocsparseio::type_t       ptr_type;
    rocsparseio::type_t       ind_type;
    rocsparseio::type_t       val_type;
    rocsparseio::index_base_t index_base;
    ROCSPARSEIO_C_CHECK(rocsparseio::read_metadata_sparse_gebsx(handle,
                                                                &dir,
                                                                &dirb,
                                                                mb,
                                                                nb,
                                                                nnzb,
                                                                row_block_dim,
                                                                col_block_dim,
                                                                &ptr_type,
                                                                &ind_type,
                                                                &val_type,
                                                                &index_base));
    dir_[0]      = (rocsparseio_direction)dir;
    dirb_[0]     = (rocsparseio_direction)dirb;
    ptr_type_[0] = (rocsparseio_type)ptr_type;
    ind_type_[0] = (rocsparseio_type)ind_type;
    val_type_[0] = (rocsparseio_type)val_type;
    base_[0]     = (rocsparseio_index_base)index_base;
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status
    rocsparseiox_read_sparse_gebsx(rocsparseio_handle handle, void* ptr, void* ind, void* val)
{
    ROCSPARSEIO_C_CHECK_ARG(0 == handle, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK(rocsparseio::read_sparse_gebsx(handle, ptr, ind, val));
    return rocsparseio_status_success;
}

extern "C" rocsparseio_status rocsparseio_read_sparse_gebsx(rocsparseio_handle handle,

                                                            rocsparseio_direction* dir,
                                                            rocsparseio_direction* dirb,

                                                            uint64_t* mb,
                                                            uint64_t* nb,
                                                            uint64_t* nnzb,

                                                            uint64_t* row_block_dim,
                                                            uint64_t* col_block_dim,

                                                            rocsparseio_type* ptr_type,
                                                            void**            ptr,

                                                            rocsparseio_type* ind_type,
                                                            void**            ind,

                                                            rocsparseio_type*       val_type,
                                                            void**                  val,
                                                            rocsparseio_index_base* base)
{
    ROCSPARSEIO_C_CHECK_ARG(handle == 0, rocsparseio::status_t::invalid_handle);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dir, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == dirb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == mb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == nnzb, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == row_block_dim, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == col_block_dim, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val_type, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ptr, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == ind, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == val, rocsparseio::status_t::invalid_pointer);
    ROCSPARSEIO_C_CHECK_ARG(nullptr == base, rocsparseio::status_t::invalid_pointer);

    ROCSPARSEIO_C_CHECK(rocsparseiox_read_metadata_sparse_gebsx(handle,
                                                                dir,
                                                                dirb,
                                                                mb,
                                                                nb,
                                                                nnzb,
                                                                row_block_dim,
                                                                col_block_dim,
                                                                ptr_type,
                                                                ind_type,
                                                                val_type,
                                                                base));

    rocsparseio::direction_t direction = dir[0];

    uint64_t ptr_size = 0;
    switch(direction)
    {
    case rocsparseio::direction_t::row:
    {
        ptr_size = mb[0] + 1;
        break;
    }
    case rocsparseio::direction_t::column:
    {
        ptr_size = nb[0] + 1;
        break;
    }
    }

    rocsparseio::type_t ptr_type_ = ptr_type[0];
    ptr[0]                        = malloc(ptr_type_.size() * ptr_size);
    // LCOV_EXCL_START
    if(!ptr[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    rocsparseio::type_t ind_type_ = ind_type[0];
    ind[0]                        = malloc(ind_type_.size() * (nnzb[0]));
    // LCOV_EXCL_START
    if(!ind[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    rocsparseio::type_t val_type_ = val_type[0];
    val[0] = malloc(val_type_.size() * (nnzb[0] * row_block_dim[0] * col_block_dim[0]));
    // LCOV_EXCL_START
    if(!val[0])
    {
        return rocsparseio_status_invalid_memory;
    }
    // LCOV_EXCL_STOP

    return rocsparseiox_read_sparse_gebsx(handle, ptr[0], ind[0], val[0]);
}
