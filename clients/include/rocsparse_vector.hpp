/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_VECTOR_HPP
#define ROCSPARSE_VECTOR_HPP

#include "rocsparse_init.hpp"

#include <cinttypes>
#include <locale.h>
struct memory_mode
{
    typedef enum _value
    {
        device = 0,
        host,
        managed
    } value_t;
};

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, memory_mode::value_t MODE, size_t PAD, typename U>
class d_vector
{
protected:
    size_t m_size{}, m_bytes{};

#ifdef GOOGLE_TEST
    U guard[PAD];
    d_vector(size_t s)
        : m_size(s)
        , m_bytes((s + PAD * 2) * sizeof(T))
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            rocsparse_init_nan(guard, PAD);
        }
    }

    d_vector()
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            rocsparse_init_nan(guard, PAD);
        }
    }

    void reset(size_t s)
    {
        this->m_size  = s;
        this->m_bytes = (s + PAD * 2) * sizeof(T);
    }

#else
    d_vector(size_t s)
        : m_size(s)
        , m_bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }

    d_vector() {}

    void reset(size_t s)
    {
        this->m_size  = s;
        this->m_bytes = s * sizeof(T);
    }

#endif

    T* device_vector_setup()
    {

        if(this->m_size == 0)
        {
            return (T*)0x4;
        }
        T* d;
        switch(MODE)
        {
        case memory_mode::device:
        {
            if((hipMalloc)(&d, this->m_bytes) != hipSuccess)
            {
                fprintf(stderr,
                        "Error allocating %'zu bytes (%zu GB)\n",
                        this->m_bytes,
                        this->m_bytes >> 30);
                d = nullptr;
            }
#ifdef GOOGLE_TEST
            else
            {
                if(PAD > 0)
                {
                    // Copy guard to device memory before allocated memory
                    hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                    // Point to allocated block
                    d += PAD;

                    // Copy guard to device memory after allocated memory
                    hipMemcpy(d + this->m_size, guard, sizeof(guard), hipMemcpyHostToDevice);
                }
            }
#endif

            break;
        }
        case memory_mode::managed:
        {

            if((hipMallocManaged)(&d, this->m_bytes) != hipSuccess)
            {
                fprintf(stderr,
                        "Error allocating %'zu bytes (%zu GB)\n",
                        this->m_bytes,
                        this->m_bytes >> 30);
                d = nullptr;
            }
#ifdef GOOGLE_TEST
            else
            {
                if(PAD > 0)
                {
                    // Copy guard to device memory before allocated memory
                    hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                    // Point to allocated block
                    d += PAD;

                    // Copy guard to device memory after allocated memory
                    hipMemcpy(d + this->m_size, guard, sizeof(guard), hipMemcpyHostToDevice);
                }
            }
#endif

            break;
        }
        }
        return d;
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr && d != ((T*)0x4))
        {
#ifdef GOOGLE_TEST
            if(PAD > 0)
            {
                U host[PAD];

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d + this->m_size, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses a batch of device memory pointers and
            an array of pointers in host memory*/
template <typename T, size_t PAD = 4096, typename U = T>
class device_batch_vector : private d_vector<T, memory_mode::device, PAD, U>
{
public:
    explicit device_batch_vector(size_t b, size_t s)
        : batch(b)
        , d_vector<T, memory_mode::device, PAD, U>(s)
    {
        data = (T**)malloc(batch * sizeof(T*));
        for(int b = 0; b < batch; ++b)
            data[b] = this->device_vector_setup();
    }

    ~device_batch_vector()
    {
        if(data != nullptr)
        {
            for(int b = 0; b < batch; ++b)
                this->device_vector_teardown(data[b]);
            free(data);
        }
    }

    T* operator[](int n)
    {
        return data[n];
    }

    operator T**()
    {
        return data;
    }

    // Disallow copying or assigning
    device_batch_vector(const device_batch_vector&) = delete;
    device_batch_vector& operator=(const device_batch_vector&) = delete;

private:
    T**    data;
    size_t batch;
};

template <typename T,
          memory_mode::value_t MODE = memory_mode::device,
          size_t               PAD  = 4096,
          typename U                = T>
class device_vector;

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses host memory */
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return this->data();
    }
    operator const T*() const
    {
        return this->data();
    }
    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const device_vector<T, THAT_MODE>& that);
    void transfer_from(const host_vector<T>& that);
};

constexpr hipMemcpyKind get_transfer_mode(memory_mode::value_t MODE, memory_mode::value_t THAT_MODE)
{
    switch(MODE)
    {
    case memory_mode::host:
    {
        switch(THAT_MODE)
        {
        case memory_mode::host:
        {
            return hipMemcpyHostToHost;
        }
        case memory_mode::device:
        {
            return hipMemcpyDeviceToHost;
        }
        case memory_mode::managed:
        {
            return hipMemcpyHostToHost;
        }
        }
    }
    case memory_mode::device:
    {
        switch(THAT_MODE)
        {
        case memory_mode::host:
        {
            return hipMemcpyHostToDevice;
        }
        case memory_mode::device:
        {
            return hipMemcpyDeviceToDevice;
        }
        case memory_mode::managed:
        {
            return hipMemcpyDeviceToDevice;
        }
        }
    }
    case memory_mode::managed:
    {
        switch(THAT_MODE)
        {
        case memory_mode::host:
        {
            return hipMemcpyHostToHost;
        }
        case memory_mode::managed:
        {
            return hipMemcpyHostToHost;
        }
        case memory_mode::device:
        {
            return hipMemcpyDeviceToDevice;
        }
        }
    }
    }
}

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses device memory */
template <typename T, memory_mode::value_t MODE, size_t PAD, typename U>
class device_vector : private d_vector<T, MODE, PAD, U>
{
public:
    device_vector() {}

    // Must wrap constructor and destructor in functions to allow Google Test macros to work
    explicit device_vector(size_t s)
        : d_vector<T, MODE, PAD, U>(s)
    {
        this->data = this->device_vector_setup();
    }

    explicit device_vector(const host_vector<T>& that, bool transfer_from_host = true)
        : d_vector<T, MODE, PAD, U>(that.size())
    {
        this->data = this->device_vector_setup();
        if(transfer_from_host)
        {
            this->transfer_from(that);
        }
    }

    size_t size() const
    {
        return this->m_size;
    }

    void resize(size_t s)
    {
        if(s != this->m_size)
        {
            this->device_vector_teardown(this->data);
            this->reset(s);
            this->data = this->device_vector_setup();
        }
    }

    ~device_vector()
    {
        this->device_vector_teardown(this->data);
        this->data = nullptr;
    }

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return this->data;
    }

    operator const T*() const
    {
        return this->data;
    }

    // Tell whether malloc failed
    explicit operator bool() const
    {
        return this->data != nullptr;
    }

    // Disallow copying or assigning
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;

    template <memory_mode::value_t THAT_MODE>
    void transfer_from(const device_vector<T, THAT_MODE>& that)
    {
        CHECK_HIP_ERROR(this->m_size == that.size() ? hipSuccess : hipErrorInvalidValue);
        CHECK_HIP_ERROR(hipMemcpy(this->data,
                                  (const T*)that,
                                  sizeof(T) * that.size(),
                                  get_transfer_mode(MODE, THAT_MODE)));
    }

    void transfer_from(const host_vector<T>& that)
    {
        CHECK_HIP_ERROR(this->m_size == that.size() ? hipSuccess : hipErrorInvalidValue);
        CHECK_HIP_ERROR(hipMemcpy(this->data,
                                  (const T*)that,
                                  sizeof(T) * that.size(),
                                  get_transfer_mode(MODE, memory_mode::host)));
    }

private:
    T* data{};
};

template <typename T>
template <memory_mode::value_t THAT_MODE>
void host_vector<T>::transfer_from(const device_vector<T, THAT_MODE>& that)
{
    CHECK_HIP_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(this->data(),
                              (const T*)that,
                              sizeof(T) * that.size(),
                              get_transfer_mode(memory_mode::host, THAT_MODE)));
}

template <typename T>
void host_vector<T>::transfer_from(const host_vector<T>& that)
{
    CHECK_HIP_ERROR(this->size() == that.size() ? hipSuccess : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(this->data(),
                              (const T*)that,
                              sizeof(T) * that.size(),
                              get_transfer_mode(memory_mode::host, memory_mode::host)));
}

template <memory_mode::value_t mode_>
struct memory_traits;

template <>
struct memory_traits<memory_mode::device>
{
    template <typename S>
    using array_t = device_vector<S>;
};

template <>
struct memory_traits<memory_mode::managed>
{
    template <typename S>
    using array_t = device_vector<S, memory_mode::managed>;
};

template <>
struct memory_traits<memory_mode::host>
{
    template <typename S>
    using array_t = host_vector<S>;
};

#endif // ROCSPARSE_VECTOR_HPP
