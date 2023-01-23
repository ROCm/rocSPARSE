/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_const_spmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_const_spmat_descr A{};
    int64_t                     rows           = safe_size;
    int64_t                     cols           = safe_size;
    int64_t                     nnz            = safe_size;
    rocsparse_direction         bell_block_dir = rocsparse_direction_row;
    int64_t                     bell_block_dim = safe_size;
    int64_t                     bell_cols      = safe_size;
    const void*                 row_data       = (const void*)0x4;
    const void*                 col_data       = (const void*)0x4;
    const void*                 val_data       = (const void*)0x4;
    rocsparse_index_base        base           = rocsparse_index_base_zero;
    rocsparse_format            format         = rocsparse_format_csr;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

#define PARAMS_CREATE_COO &A, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_coo_descr, PARAMS_CREATE_COO);
#undef PARAMS_CREATE_COO

#define PARAMS_CREATE_CSR \
    &A, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_csr_descr, PARAMS_CREATE_CSR);
#undef PARAMS_CREATE_CSR

#define PARAMS_CREATE_CSC \
    &A, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_csc_descr, PARAMS_CREATE_CSC);
#undef PARAMS_CREATE_CSC

#define PARAMS_CREATE_ELL                                                                       \
    &A, rows, cols, bell_block_dir, bell_block_dim, bell_cols, col_data, val_data, itype, base, \
        ttype
    auto_testing_bad_arg(rocsparse_create_const_bell_descr, PARAMS_CREATE_ELL);
#undef PARAMS_CREATE_ELL

    // rocsparse_destroy_spmat_descr_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, 0, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, rows, 0, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, rows, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, 0, cols, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, 0, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, cols, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, cols, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, rows, 0, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, 0, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, rows, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&A,
                                                              0,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              bell_cols,
                                                              nullptr,
                                                              nullptr,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_bell_descr(
            &A, rows, 0, bell_block_dir, bell_block_dim, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&A,
                                                              rows,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              0,
                                                              nullptr,
                                                              nullptr,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    rocsparse_const_spmat_descr coo;
    rocsparse_const_spmat_descr csr;
    rocsparse_const_spmat_descr csc;
    rocsparse_const_spmat_descr ell;

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_coo_descr(
            &coo, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &csr, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &csc, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&ell,
                                                              rows,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              bell_cols,
                                                              col_data,
                                                              val_data,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);

    const void** row_data_ptr = (const void**)0x4;
    const void** col_data_ptr = (const void**)0x4;
    const void** val_data_ptr = (const void**)0x4;

#define PARAMS_GET_COO \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_coo_get, PARAMS_GET_COO);
#undef PARAMS_GET_COO

#define PARAMS_GET_CSR \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &jtype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_csr_get, PARAMS_GET_CSR);
#undef PARAMS_GET_CSR

#define PARAMS_GET_ELL                                                                         \
    A, &rows, &cols, &bell_block_dir, &bell_block_dim, &bell_cols, col_data_ptr, val_data_ptr, \
        &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_bell_get, PARAMS_GET_ELL);
#undef PARAMS_GET_ELL

    // rocsparse_spmat_get_size_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(nullptr, &rows, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_format_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(nullptr, &format),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_index_base_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_const_spmat_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(nullptr, (const void**)&val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    int batch_count = safe_size;

    // rocsparse_spmat_get_strided_batch_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(nullptr, &batch_count),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);
}
template <typename I, typename J, typename T>
void testing_const_spmat_descr(const Arguments& arg)
{
    int64_t              m         = arg.M;
    int64_t              n         = arg.N;
    int64_t              nnz       = arg.nnz;
    rocsparse_direction  block_dir = arg.direction;
    int64_t              block_dim = arg.block_dim;
    rocsparse_index_base base      = arg.baseA;

    int64_t col_width = m;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        return;
    }

    device_vector<I> row_data(nnz);
    device_vector<I> col_data(nnz);
    device_vector<J> idx_data(m + 1);
    device_vector<T> val_data(nnz);

    if(arg.unit_check)
    {
        /*
        *   COO
        */
        rocsparse_const_spmat_descr coo;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_coo_descr(
            &coo, m, n, nnz, row_data, col_data, val_data, itype, base, ttype));

        int64_t              coo_m;
        int64_t              coo_n;
        int64_t              coo_nnz;
        rocsparse_index_base coo_base;
        rocsparse_indextype  coo_itype;
        rocsparse_datatype   coo_ttype;
        const T*             coo_val;
        const I*             coo_row;
        const I*             coo_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_get(coo,
                                                      &coo_m,
                                                      &coo_n,
                                                      &coo_nnz,
                                                      (const void**)&coo_row,
                                                      (const void**)&coo_col,
                                                      (const void**)&coo_val,
                                                      &coo_itype,
                                                      &coo_base,
                                                      &coo_ttype));

        unit_check_scalar(m, coo_m);
        unit_check_scalar(n, coo_n);
        unit_check_scalar(nnz, coo_nnz);
        unit_check_enum(base, coo_base);
        unit_check_enum(itype, coo_itype);
        unit_check_enum(ttype, coo_ttype);
        ASSERT_EQ(val_data, coo_val);
        ASSERT_EQ(row_data, coo_row);
        ASSERT_EQ(col_data, coo_col);

        coo_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(coo, (const void**)&coo_val));
        ASSERT_EQ(val_data, coo_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(coo));

        /*
        *   CSR
        */
        rocsparse_const_spmat_descr csr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_csr_descr(
            &csr, m, n, nnz, idx_data, col_data, val_data, jtype, itype, base, ttype));

        int64_t              csr_m;
        int64_t              csr_n;
        int64_t              csr_nnz;
        rocsparse_index_base csr_base;
        rocsparse_indextype  csr_itype;
        rocsparse_indextype  csr_jtype;
        rocsparse_datatype   csr_ttype;
        const T*             csr_val;
        const J*             csr_idx;
        const I*             csr_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_csr_get(csr,
                                                      &csr_m,
                                                      &csr_n,
                                                      &csr_nnz,
                                                      (const void**)&csr_idx,
                                                      (const void**)&csr_col,
                                                      (const void**)&csr_val,
                                                      &csr_jtype,
                                                      &csr_itype,
                                                      &csr_base,
                                                      &csr_ttype));

        unit_check_scalar(m, csr_m);
        unit_check_scalar(n, csr_n);
        unit_check_scalar(nnz, csr_nnz);
        unit_check_enum(base, csr_base);
        unit_check_enum(itype, csr_itype);
        unit_check_enum(jtype, csr_jtype);
        unit_check_enum(ttype, csr_ttype);
        ASSERT_EQ(val_data, csr_val);
        ASSERT_EQ(idx_data, csr_idx);
        ASSERT_EQ(col_data, csr_col);

        csr_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(csr, (const void**)&csr_val));
        ASSERT_EQ(val_data, csr_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(csr));

        /*
        *   CSC
        */
        rocsparse_const_spmat_descr csc;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_csc_descr(
            &csc, n, m, nnz, idx_data, row_data, val_data, jtype, itype, base, ttype));

        int64_t              csc_m;
        int64_t              csc_n;
        int64_t              csc_nnz;
        rocsparse_index_base csc_base;
        rocsparse_indextype  csc_itype;
        rocsparse_indextype  csc_jtype;
        rocsparse_datatype   csc_ttype;
        const T*             csc_val;
        const J*             csc_idx;
        const I*             csc_row;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_csc_get(csc,
                                                      &csc_n,
                                                      &csc_m,
                                                      &csc_nnz,
                                                      (const void**)&csc_idx,
                                                      (const void**)&csc_row,
                                                      (const void**)&csc_val,
                                                      &csc_jtype,
                                                      &csc_itype,
                                                      &csc_base,
                                                      &csc_ttype));

        unit_check_scalar(m, csc_m);
        unit_check_scalar(n, csc_n);
        unit_check_scalar(nnz, csc_nnz);
        unit_check_enum(base, csc_base);
        unit_check_enum(itype, csc_itype);
        unit_check_enum(jtype, csc_jtype);
        unit_check_enum(ttype, csc_ttype);
        ASSERT_EQ(val_data, csc_val);
        ASSERT_EQ(idx_data, csc_idx);
        ASSERT_EQ(row_data, csc_row);

        csc_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(csc, (const void**)&csc_val));
        ASSERT_EQ(val_data, csc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(csc));

        /*
        *   BELL
        */
        rocsparse_const_spmat_descr bell;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_bell_descr(
            &bell, m, n, block_dir, block_dim, col_width, col_data, val_data, itype, base, ttype));

        int64_t              bell_m;
        int64_t              bell_n;
        int64_t              bell_col_width;
        int64_t              bell_block_dim;
        rocsparse_direction  bell_block_dir;
        rocsparse_index_base bell_base;
        rocsparse_indextype  bell_itype;
        rocsparse_datatype   bell_ttype;
        const T*             bell_val;
        const I*             bell_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_bell_get(bell,
                                                       &bell_m,
                                                       &bell_n,
                                                       &bell_block_dir,
                                                       &bell_block_dim,
                                                       &bell_col_width,
                                                       (const void**)&bell_col,
                                                       (const void**)&bell_val,
                                                       &bell_itype,
                                                       &bell_base,
                                                       &bell_ttype));

        unit_check_scalar(m, bell_m);
        unit_check_scalar(n, bell_n);
        unit_check_scalar(col_width, bell_col_width);
        unit_check_scalar(block_dim, bell_block_dim);
        unit_check_enum(block_dir, bell_block_dir);
        unit_check_enum(base, bell_base);
        unit_check_enum(itype, bell_itype);
        unit_check_enum(ttype, bell_ttype);
        ASSERT_EQ(val_data, bell_val);
        ASSERT_EQ(col_data, bell_col);

        bell_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(bell, (const void**)&bell_val));
        ASSERT_EQ(val_data, bell_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(bell));
    }

    if(arg.timing)
    {
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                        \
    template void testing_const_spmat_descr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_const_spmat_descr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_const_spmat_descr_extra(const Arguments& arg) {}
