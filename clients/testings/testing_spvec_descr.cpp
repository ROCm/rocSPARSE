/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

void testing_spvec_descr_bad_arg(const Arguments& arg)
{
    int64_t size = 100;
    int64_t nnz  = 100;

    rocsparse_indextype  itype = rocsparse_indextype_i32;
    rocsparse_datatype   ttype = rocsparse_datatype_f32_r;
    rocsparse_index_base base  = rocsparse_index_base_zero;

    // Allocate memory on device
    device_vector<int>   idx_data(100);
    device_vector<float> val_data(100);

    if(!idx_data || !val_data)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_spvec_descr x;

    // rocsparse_create_spvec_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(nullptr, size, nnz, idx_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, -1, nnz, idx_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, -1, idx_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, nnz, nullptr, val_data, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, nnz, idx_data, nullptr, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, 100, 200, idx_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);

    // rocsparse_destroy_spvec_descr
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, 0, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);

    // Create valid descriptor
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_spvec_descr(&x, size, nnz, idx_data, val_data, itype, base, ttype),
        rocsparse_status_success);

    // rocsparse_spvec_get
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            nullptr, &size, &nnz, (void**)&idx_data, (void**)&val_data, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            x, nullptr, &nnz, (void**)&idx_data, (void**)&val_data, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            x, &size, nullptr, (void**)&idx_data, (void**)&val_data, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(x, &size, &nnz, nullptr, (void**)&val_data, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(x, &size, &nnz, (void**)&idx_data, nullptr, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            x, &size, &nnz, (void**)&idx_data, (void**)&val_data, nullptr, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            x, &size, &nnz, (void**)&idx_data, (void**)&val_data, &itype, nullptr, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvec_get(
            x, &size, &nnz, (void**)&idx_data, (void**)&val_data, &itype, &base, nullptr),
        rocsparse_status_invalid_pointer);

    // rocsparse_spvec_get_index_base
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_index_base(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spvec_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_values(nullptr, (void**)&val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_get_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spvec_set_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_set_values(nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvec_set_values(x, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
}
