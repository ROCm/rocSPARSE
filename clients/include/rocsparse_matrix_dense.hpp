/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_MATRIX_DENSE_HPP
#define ROCSPARSE_MATRIX_DENSE_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE, typename T>
struct dense_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;
    rocsparse_int   m{};
    rocsparse_int   n{};
    rocsparse_int   ld{};
    array_t<T>      val{};
    rocsparse_order order;
    dense_matrix(){};
    ~dense_matrix(){};

    dense_matrix(rocsparse_int   m_,
                 rocsparse_int   n_,
                 rocsparse_order order_ = rocsparse_order_column)
        : m(m_)
        , n(n_)
        , ld((order_ == rocsparse_order_column) ? m_ : n_)
        , val(m_ * n_)
        , order(order_){};

    void print() const
    {
        switch(MODE)
        {
        case memory_mode::host:
        case memory_mode::managed:
        {
            switch(this->order)
            {
            case rocsparse_order_column:
            {
                for(int i = 0; i < this->m; ++i)
                {
                    for(int j = 0; j < this->n; ++j)
                    {
                        std::cout << " " << this->val[j * this->ld + i];
                    }
                    std::cout << std::endl;
                }
                break;
            }
            case rocsparse_order_row:
            {
                for(int i = 0; i < this->m; ++i)
                {
                    for(int j = 0; j < this->n; ++j)
                    {
                        std::cout << " " << this->val[i * this->ld + j];
                    }
                    std::cout << std::endl;
                }
                break;
            }
            }
            break;
        }
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T> on_host(*this);
            on_host.print();
            break;
        }
        }
    };

    operator T*()
    {
        return this->val;
    }

    operator const T*() const
    {
        return this->val;
    }

    dense_matrix(const dense_matrix<MODE, T>& that, bool transfer = true)
        : m(that.m)
        , n(that.n)
        , ld((that.order == rocsparse_order_column) ? that.m : that.n)
        , val(that.m * that.n)
        , order(that.order)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    dense_matrix(const dense_matrix<THAT_MODE, T>& that, bool transfer = true)
        : m(that.m)
        , n(that.n)
        , ld((that.order == rocsparse_order_column) ? that.m : that.n)
        , val(that.m * that.n)
        , order(that.order)
    {
        if(transfer)
        {
            this->transfer_from(that);
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const dense_matrix<THAT_MODE, T>& that_)
    {
        CHECK_HIP_ERROR((this->m == that_.m && this->n == that_.n) ? hipSuccess
                                                                   : hipErrorInvalidValue);
        CHECK_HIP_ERROR((this->order == that_.order) ? hipSuccess : hipErrorInvalidValue);
        //
        // FOR NOW RESTRUCTION..
        //
        switch(that_.order)
        {
        case rocsparse_order_row:
        {
            CHECK_HIP_ERROR((that_.n == that_.ld) ? hipSuccess : hipErrorInvalidValue);
            CHECK_HIP_ERROR((this->n == this->ld) ? hipSuccess : hipErrorInvalidValue);
            break;
        }
        case rocsparse_order_column:
        {
            CHECK_HIP_ERROR((that_.m == that_.ld) ? hipSuccess : hipErrorInvalidValue);
            CHECK_HIP_ERROR((this->m == this->ld) ? hipSuccess : hipErrorInvalidValue);
            break;
        }
        }
        this->val.transfer_from(that_.val);
    }

    template <memory_mode::value_t THAT_MODE>
    void unit_check(const dense_matrix<THAT_MODE, T>& that_) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T> on_host(*this);
            on_host.unit_check(that_);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::managed:
            case memory_mode::host:
            {
                unit_check_general<rocsparse_int>(1, 1, 1, &this->m, &that_.m);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->n, &that_.n);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->ld, &that_.ld);

                {
                    rocsparse_int a = (rocsparse_int)this->order;
                    rocsparse_int b = (rocsparse_int)that_.order;
                    unit_check_general<rocsparse_int>(1, 1, 1, &a, &b);
                }

                switch(this->order)
                {
                case rocsparse_order_column:
                {
                    unit_check_general<T>(this->m, this->n, this->ld, this->val, that_.val);
                    break;
                }

                case rocsparse_order_row:
                {
                    //
                    // Little trick
                    // If this poses a problem, we need to refactor unit_check_general.
                    //
                    unit_check_general<T>(this->n, this->m, this->ld, this->val, that_.val);
                    break;
                }
                }
                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T> that(that_);
                this->unit_check(that);
                break;
            }
            }
            break;
        }
        }
    }

    template <memory_mode::value_t THAT_MODE>
    void near_check(const dense_matrix<THAT_MODE, T>& that_,
                    floating_data_t<T>                tol = default_tolerance<T>::value) const
    {
        switch(MODE)
        {
        case memory_mode::device:
        {
            dense_matrix<memory_mode::host, T> on_host(*this);
            on_host.near_check(that_, tol);
            break;
        }

        case memory_mode::managed:
        case memory_mode::host:
        {
            switch(THAT_MODE)
            {
            case memory_mode::host:
            case memory_mode::managed:
            {
                unit_check_general<rocsparse_int>(1, 1, 1, &this->m, &that_.m);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->n, &that_.n);
                unit_check_general<rocsparse_int>(1, 1, 1, &this->ld, &that_.ld);

                {
                    rocsparse_int a = (rocsparse_int)this->order;
                    rocsparse_int b = (rocsparse_int)that_.order;
                    unit_check_general<rocsparse_int>(1, 1, 1, &a, &b);
                }

                switch(this->order)
                {
                case rocsparse_order_column:
                {
                    near_check_general<T>(this->m, this->n, this->ld, this->val, that_.val, tol);
                    break;
                }

                case rocsparse_order_row:
                {
                    //
                    // Little trick
                    // If this poses a problem, we need to refactor unit_check_general.
                    //
                    near_check_general<T>(this->n, this->m, this->ld, this->val, that_.val, tol);
                    break;
                }
                }

                break;
            }
            case memory_mode::device:
            {
                dense_matrix<memory_mode::host, T> that(that_);
                this->near_check(that, tol);
                break;
            }
            }
            break;
        }
        }
    }
};

template <typename T>
using host_dense_matrix = dense_matrix<memory_mode::host, T>;

template <typename T>
using device_dense_matrix = dense_matrix<memory_mode::device, T>;

template <typename T>
using managed_dense_matrix = dense_matrix<memory_mode::managed, T>;

#endif // ROCSPARSE_MATRIX_DENSE_HPP
