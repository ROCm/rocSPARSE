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

template <typename I, typename T>
void testing_const_spvec_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_const_spvec_descr x{};
    int64_t                     size     = safe_size;
    int64_t                     nnz      = safe_size;
    const void*                 val_data = (void*)0x4;
    const void*                 idx_data = (void*)0x4;
    rocsparse_index_base        base     = rocsparse_index_base_zero;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

#define PARAMS_CREATE &x, size, nnz, idx_data, val_data, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_spvec_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

    // rocsparse_destroy_spvec_descr_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_spvec_descr(&x, 0, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_spvec_descr(&x, size, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_spvec_descr(&x, size, nnz, idx_data, val_data, itype, base, ttype),
        rocsparse_status_success);

    const void** val_data_ptr = (const void**)0x4;
    const void** idx_data_ptr = (const void**)0x4;

#define PARAMS_GET x, &size, &nnz, idx_data_ptr, val_data_ptr, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_spvec_get, PARAMS_GET);
#undef PARAMS_GET

    // rocsparse_spvec_get_index_base
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spvec_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spvec_get_values(nullptr, val_data_ptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spvec_get_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
}

template <typename I, typename T>
void testing_const_spvec_descr(const Arguments& arg)
{
    int64_t              m     = arg.M;
    int64_t              nnz   = arg.nnz;
    rocsparse_index_base base  = arg.baseA;
    rocsparse_indextype  itype = get_indextype<I>();
    rocsparse_datatype   ttype = get_datatype<T>();

    device_vector<T> values(nnz);
    device_vector<I> idx_data(nnz);

    if(arg.unit_check)
    {
        rocsparse_const_spvec_descr A{};

        // Create valid descriptor
        CHECK_ROCSPARSE_ERROR(
            rocsparse_create_const_spvec_descr(&A, m, nnz, idx_data, values, itype, base, ttype));

        int64_t              gold_m;
        int64_t              gold_nnz;
        rocsparse_index_base gold_base;
        rocsparse_datatype   gold_ttype;
        rocsparse_indextype  gold_itype;

        const T* gold_values;
        const I* gold_idx_data;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_spvec_get(A,
                                                        &gold_m,
                                                        &gold_nnz,
                                                        (const void**)&gold_idx_data,
                                                        (const void**)&gold_values,
                                                        &gold_itype,
                                                        &gold_base,
                                                        &gold_ttype));

        unit_check_scalar<int64_t>(m, gold_m);
        unit_check_scalar<int64_t>(nnz, gold_nnz);
        unit_check_enum<rocsparse_datatype>(ttype, gold_ttype);
        unit_check_enum<rocsparse_indextype>(itype, gold_itype);
        unit_check_enum<rocsparse_index_base>(base, gold_base);

        ASSERT_EQ(values, (const T*)gold_values);
        ASSERT_EQ(idx_data, (const I*)gold_idx_data);

        gold_values = nullptr;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_spvec_get_values(A, (const void**)&gold_values));

        ASSERT_EQ(values, (const T*)gold_values);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spvec_descr(A));
    }

    if(arg.timing)
    {
    }
}

#define INSTANTIATE(ITYPE, TTYPE)                                                        \
    template void testing_const_spvec_descr_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_const_spvec_descr<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_const_spvec_descr_extra(const Arguments& arg) {}
