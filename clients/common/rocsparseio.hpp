/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define ROCSPARSEIO_CHECK(_todo)                                                                \
    {                                                                                           \
        rocsparseio::status_t check_status = _todo;                                             \
        if(check_status != rocsparseio::status_t::success)                                      \
        {                                                                                       \
            fprintf(stderr, "ROCSPARSEIO_CHECK FAILED, LINE %d FILE %s\n", __LINE__, __FILE__); \
            return check_status;                                                                \
        }                                                                                       \
    }                                                                                           \
    (void)0

#define ROCSPARSEIO_CHECK_ARG(check_arg_cond, check_arg_status) \
    if(check_arg_cond)                                          \
    {                                                           \
        return check_arg_status;                                \
    }                                                           \
    (void)0

#define ROCSPARSEIO_C_CHECK(_todo)                         \
    {                                                      \
        rocsparseio::status_t check_status = _todo;        \
        if(check_status != rocsparseio::status_t::success) \
        {                                                  \
            fprintf(stderr, "ROCSPARSEIO_CHECK FAILED\n"); \
            return (rocsparseio_status)check_status;       \
        }                                                  \
    }                                                      \
    (void)0

#define ROCSPARSEIO_C_CHECK_ARG(check_arg_cond, check_arg_status) \
    if(check_arg_cond)                                            \
    {                                                             \
        return (rocsparseio_status)check_arg_status;              \
    }

#include "rocsparseio.h"

namespace rocsparseio
{

#define ROCSPARSEIO_ENUM_EACH_STATUS                \
    ROCSPARSEIO_ENUM_STATUS(success)                \
    ROCSPARSEIO_ENUM_STATUS(invalid_handle)         \
    ROCSPARSEIO_ENUM_STATUS(invalid_pointer)        \
    ROCSPARSEIO_ENUM_STATUS(invalid_value)          \
    ROCSPARSEIO_ENUM_STATUS(invalid_enum)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_file)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_file_operation) \
    ROCSPARSEIO_ENUM_STATUS(invalid_format)         \
    ROCSPARSEIO_ENUM_STATUS(invalid_mode)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_size)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_memory)

    //!
    //! @brief c++11 struct for status enum.
    //!
    struct status_t
    {

#define ROCSPARSEIO_ENUM_STATUS(x_) x_ = rocsparseio_status_##x_,

        typedef enum value_type_ : rocsparseio_status_t
        {
            ROCSPARSEIO_ENUM_EACH_STATUS
        } value_type;

#undef ROCSPARSEIO_ENUM_STATUS

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_status() const
        {
            return (rocsparseio_status)this->value;
        }

        inline constexpr status_t() {}
        inline constexpr status_t(rocsparseio_status_t ival)
            : value((value_type)ival)
        {
        }

        inline constexpr status_t(rocsparseio_status ival)
            : value((value_type)ival)
        {
        }

        inline bool is_invalid() const
        {
#define ROCSPARSEIO_ENUM_STATUS(x_) case x_:
            switch(this->value)
            {
                ROCSPARSEIO_ENUM_EACH_STATUS
                {
                    return false;
                }
            }
#undef ROCSPARSEIO_ENUM_STATUS
            return true;
        }

        inline const char* to_string() const
        {
#define ROCSPARSEIO_ENUM_STATUS(x_) \
    case x_:                        \
    {                               \
        return #x_;                 \
    }
            switch(this->value)
            {
                ROCSPARSEIO_ENUM_EACH_STATUS;
            }
#undef ROCSPARSEIO_ENUM_STATUS
            return "rocsparseio-unknown";
        }

#undef ROCSPARSEIO_ENUM_EACH_STATUS
    };

    //!
    //! @brief c++11 struct for format enum.
    //!
    struct format_t
    {
        typedef enum value_type_ : rocsparseio_format_t
        {
            dense_vector = rocsparseio_format_dense_vector,
            dense_matrix = rocsparseio_format_dense_matrix,
            sparse_csx   = rocsparseio_format_sparse_csx,
            sparse_gebsx = rocsparseio_format_sparse_gebsx,
            sparse_coo   = rocsparseio_format_sparse_coo,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline constexpr format_t(){};
        inline constexpr format_t(rocsparseio_format_t ival)
            : value((value_type)ival)
        {
        }

        inline constexpr format_t(rocsparseio_format ival)
            : value((value_type)ival)
        {
        }

        inline explicit constexpr operator rocsparseio_format() const
        {
            return (rocsparseio_format)this->value;
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case dense_vector:
            case dense_matrix:
            case sparse_csx:
            case sparse_gebsx:
            case sparse_coo:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(dense_matrix);
                CASE(dense_vector);
                CASE(sparse_csx);
                CASE(sparse_gebsx);
                CASE(sparse_coo);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for order enum.
    //!
    struct order_t
    {
        typedef enum value_type_ : rocsparseio_order_t
        {
            row    = rocsparseio_order_row,
            column = rocsparseio_order_column,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_order() const
        {
            return (rocsparseio_order)this->value;
        }

        //    order_t(){};
        inline constexpr order_t() {}
        inline constexpr order_t(rocsparseio_order_t ival)
            : value((value_type)ival)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case row:
            case column:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(row);
                CASE(column);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for direction enum.
    //!
    struct direction_t
    {
        typedef enum value_type_ : rocsparseio_direction_t
        {
            row    = rocsparseio_direction_row,
            column = rocsparseio_direction_column,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_direction() const
        {
            return (rocsparseio_direction)this->value;
        }
        inline constexpr direction_t(){};
        inline constexpr direction_t(rocsparseio_direction_t ival)
            : value((value_type)ival)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case row:
            case column:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(row);
                CASE(column);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for type enum.
    //!
    struct type_t
    {
        typedef enum value_type_ : rocsparseio_type_t
        {
            int32     = rocsparseio_type_int32,
            int64     = rocsparseio_type_int64,
            float32   = rocsparseio_type_float32,
            float64   = rocsparseio_type_float64,
            complex32 = rocsparseio_type_complex32,
            complex64 = rocsparseio_type_complex64,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        };
        inline explicit constexpr operator rocsparseio_type() const
        {
            return (rocsparseio_type)this->value;
        };
        constexpr type_t(){};
        constexpr type_t(rocsparseio_type_t ival)
            : value((value_type)ival){};
        //    constexpr type_t(rocsparseio_type ival) : value((value_type)ival){};

        template <typename T>
        static constexpr type_t convert();

        inline size_t size() const
        {
            switch(this->value)
            {
            case int32:
            {
                return sizeof(int32_t);
            }
            case int64:
            {
                return sizeof(int64_t);
            }
            case float32:
            {
                return sizeof(float);
            }
            case float64:
            {
                return sizeof(double);
            }
            case complex32:
            {
                return sizeof(float) * 2;
            }
            case complex64:
            {
                return sizeof(double) * 2;
            }
            }
            return 0;
        };

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case int32:
            case int64:
            case float32:
            case float64:
            case complex32:
            case complex64:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(int32);
                CASE(int64);
                CASE(float32);
                CASE(float64);
                CASE(complex32);
                CASE(complex64);
#undef CASE
            }
            return "unknown";
        }
    };

    struct index_base_t
    {
        typedef enum value_index_base_ : rocsparseio_index_base_t
        {
            zero = rocsparseio_index_base_zero,
            one  = rocsparseio_index_base_one
        } value_index_base;

        value_index_base value{};
        inline constexpr operator value_index_base() const
        {
            return this->value;
        };
        inline explicit constexpr operator rocsparseio_index_base() const
        {
            return (rocsparseio_index_base)this->value;
        };
        constexpr index_base_t(){};
        constexpr index_base_t(rocsparseio_index_base_t ival)
            : value((value_index_base)ival){};
        //    constexpr index_base_t(rocsparseio_index_base ival) :
        //    value((value_index_base)ival){};

        template <typename T>
        static constexpr index_base_t convert();
        inline bool                   is_invalid() const
        {
            switch(this->value)
            {
            case one:
            case zero:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(one);
                CASE(zero);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for rwmode enum.
    //!
    struct rwmode_t
    {
        typedef enum value_type_ : rocsparseio_rwmode_t
        {
            read  = rocsparseio_rwmode_read,
            write = rocsparseio_rwmode_write
        } value_type;

        value_type value;

        inline constexpr operator value_type() const
        {
            return this->value;
        };
        inline constexpr rwmode_t(rocsparseio_rwmode_t ival)
            : value((value_type)ival)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case read:
            case write:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(read);
                CASE(write);
#undef CASE
            }
            return "unknown";
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const status_t& that_)
    {
        os << that_.to_string();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const direction_t& that_)
    {
        os << that_.to_string();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const order_t& that_)
    {
        os << that_.to_string();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const type_t& that_)
    {
        os << that_.to_string();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const rwmode_t& that_)
    {
        os << that_.to_string();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const format_t& that_)
    {
        os << that_.to_string();
        return os;
    }

} // namespace rocsparseio

struct _rocsparseio_handle
{
    rocsparseio::rwmode_t mode;
    std::string           filename{};
    FILE*                 f{};
    _rocsparseio_handle(rocsparseio::rwmode_t mode_, const char* filename_)
        : mode(mode_)
        , filename(filename_)
    {
    }
};

namespace rocsparseio
{
    using handle_t = rocsparseio_handle;

    inline status_t fread_data(FILE* in_, size_t size_, size_t nmemb_, void* data_)
    {
        if(nmemb_ != fread(data_, size_, nmemb_, in_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    template <typename source, typename target>
    inline status_t convert_scalar(const source& s, target& t)
    {
        if(!std::is_same<source, target>())
        {
            if(std::is_unsigned<source>())
            {
                if(sizeof(target) < sizeof(source))
                {
                    if(static_cast<source>(std::numeric_limits<target>::max()) < s)
                    {
                        std::cerr << "convert out of bounds " << s << ", max is "
                                  << std::numeric_limits<target>::max() << "" << std::endl;
                        return status_t::invalid_value;
                    }
                }
            }
            else
            {
                if(sizeof(target) < sizeof(source))
                {
                    if((s > static_cast<source>(std::numeric_limits<target>::max()))
                       || (s < static_cast<source>(std::numeric_limits<target>::lowest())))
                    {
                        std::cerr << "convert out of bounds " << s << ", ["
                                  << std::numeric_limits<target>::lowest() << ", "
                                  << std::numeric_limits<target>::max() << "]" << std::endl;
                        return status_t::invalid_value;
                    }
                }
            }
        }
        t = static_cast<target>(s);
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s, format_t& t)
    {
        t = static_cast<format_t>(s);
        if(t.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s, direction_t& t)
    {
        t = static_cast<direction_t>(s);
        if(t.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s, order_t& t)
    {
        t = static_cast<order_t>(s);
        if(t.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s, type_t& t)
    {
        t = static_cast<type_t>(s);
        if(t.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename T, typename J>
    inline status_t fwrite_scalar(J scalar_, FILE* out_)
    {
        {
            T value = static_cast<T>(scalar_);
            if(1 != fwrite(&value, sizeof(T), 1, out_))
            {
                return status_t::invalid_file_operation;
            }
        }
        return status_t::success;
    };

    template <typename T, typename J>
    inline status_t fread_scalar(J& scalar_, FILE* out_)
    {
        {
            T value;
            if(1 != fread(&value, sizeof(T), 1, out_))
            {
                return status_t::invalid_file_operation;
            }
            return convert_scalar(value, scalar_);
        }
        return status_t::success;
    };

    template <typename T, typename J>
    inline status_t fread_scalar(J* scalar_, FILE* out_)
    {
        {
            T value;
            if(1 != fread(&value, sizeof(T), 1, out_))
            {
                return status_t::invalid_file_operation;
            }
            J        scalar;
            status_t status = convert_scalar(value, scalar);
            if(status != status_t::success)
            {
                return status;
            }
            scalar_[0] = scalar;
        }
        return status_t::success;
    };

    inline status_t fwrite_array(FILE* out_, size_t size_, size_t nmemb_, const void* data_)
    {
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(size_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(nmemb_, out_));
        if(nmemb_ != fwrite(data_, size_, nmemb_, out_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_array(FILE* in_, void* data_)
    {
        size_t size;
        size_t nmemb;
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(size, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nmemb, in_));
        if(nmemb != fread(data_, size, nmemb, in_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_array_metadata(FILE* in_, size_t* size_, size_t* nmemb_)
    {
        long pos = ftell(in_);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(size_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nmemb_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{

    //
    // ////////////////////////////////////////////////
    //
    inline status_t fwrite_dense_vector(
        FILE* out_, type_t data_type_, size_t data_nmemb_, const void* data_, size_t data_inc_)
    {

        ROCSPARSEIO_CHECK_ARG(out_ == nullptr, status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG(data_type_.is_invalid(), status_t::invalid_value);

        ROCSPARSEIO_CHECK_ARG(((data_nmemb_ > 0) && (data_ == nullptr)), status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(((data_inc_ > 0) && (data_ == nullptr)), status_t::invalid_size);

        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(format_t::dense_vector, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_nmemb_, out_));
        size_t data_size = data_type_.size();
        if(data_inc_ == 1)
        {
            if(data_nmemb_ != fwrite(data_, data_size, data_nmemb_, out_))
            {
                return status_t::invalid_file_operation;
            }
        }
        else
        {
            for(size_t i = 0; i < data_nmemb_; ++i)
            {
                if(1
                   != fwrite(((const char*)data_) + i * data_inc_ * data_size, data_size, 1, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
        }
        return status_t::success;
    }

    inline status_t fread_format(FILE* in, format_t* format)
    {
        ROCSPARSEIO_CHECK_ARG(in == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(format == nullptr, status_t::invalid_pointer);
        long pos = ftell(in);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in));
        if(0 != fseek(in, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    };

    inline status_t fread_metadata_dense_vector(FILE* in, type_t* type, size_t* nmemb)
    {
        ROCSPARSEIO_CHECK_ARG(in == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(type == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(nmemb == nullptr, status_t::invalid_pointer);
        long     pos = ftell(in);
        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in));
        if(format != format_t::dense_vector)
        {
            std::cerr << " wrong format, not flagged as a dense_vector. " << std::endl;
            return status_t::invalid_format;
        }

        ROCSPARSEIO_CHECK(fread_scalar<size_t>(type, in));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nmemb, in));
        if(0 != fseek(in, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    };

    inline status_t fread_dense_vector(FILE* in_, void* data_, size_t inc_)
    {
        if(0 != fseek(in_, sizeof(size_t), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }
        type_t type;
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(type, in_));
        size_t size = type.size();
        size_t nmemb;
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nmemb, in_));
        if(inc_ == 1)
        {
            return fread_data(in_, size, nmemb, data_);
        }
        else
        {
            void* buff = malloc(size * nmemb);
            ROCSPARSEIO_CHECK(fread_data(in_, size, nmemb, buff));
            size_t j = 0;
            for(size_t i = 0; i < nmemb; ++i)
            {
                memcpy(((char*)data_) + j * size, ((char*)buff) + i * size, size);
                j += inc_;
            }
            free(buff);
            return status_t::invalid_value;
        }
    }

} // namespace rocsparseio

namespace rocsparseio
{

    inline status_t fwrite_dense_matrix(FILE*       out_,
                                        order_t     order_,
                                        size_t      m_,
                                        size_t      n_,
                                        type_t      data_type_,
                                        const void* data_,
                                        size_t      data_ld_)
    {
        ROCSPARSEIO_CHECK_ARG(out_ == nullptr, status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG(order_.is_invalid(), status_t::invalid_value);

        ROCSPARSEIO_CHECK_ARG(data_type_.is_invalid(), status_t::invalid_value);

        ROCSPARSEIO_CHECK_ARG((((m_ > 0) && (n_ > 0)) && (data_ == nullptr)),
                              status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG((((m_ > 0) && (n_ > 0)) && (data_ == nullptr)),
                              status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG(((order_ == order_t::row) && (data_ld_ < n_)),
                              status_t::invalid_size);

        ROCSPARSEIO_CHECK_ARG(((order_ == order_t::column) && (data_ld_ < m_)),
                              status_t::invalid_size);

        size_t data_nmemb = m_ * n_;

        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(format_t::dense_matrix, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(order_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_nmemb, out_));

        size_t data_size = data_type_.size();
        switch(order_)
        {
        case order_t::row:
        {
            if(data_ld_ == n_)
            {
                if(data_nmemb != fwrite(data_, data_size, data_nmemb, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
            else
            {
                for(size_t i = 0; i < m_; ++i)
                {
                    if(n_
                       != fwrite(
                           ((const char*)data_) + i * data_ld_ * data_size, data_size, n_, out_))
                    {
                        return status_t::invalid_file_operation;
                    }
                }
            }
            break;
        }

        case order_t::column:
        {
            if(data_ld_ == m_)
            {
                if(data_nmemb != fwrite(data_, data_size, data_nmemb, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
            else
            {
                for(size_t j = 0; j < n_; ++j)
                {
                    if(m_
                       != fwrite(
                           ((const char*)data_) + j * data_ld_ * data_size, data_size, m_, out_))
                    {
                        return status_t::invalid_file_operation;
                    }
                }
            }
            break;
        }
        }
        return status_t::success;
    };

    inline status_t fread_dense_matrix(FILE* in_, void* data_)
    {
        if(0 != fseek(in_, sizeof(size_t) * 4, SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }
        type_t type;
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(type, in_));
        size_t size = type.size();
        size_t nmemb;
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nmemb, in_));
        return fread_data(in_, size, nmemb, data_);
    }

    inline status_t fread_metadata_dense_matrix(
        FILE* in_, order_t* order_, size_t* m_, size_t* n_, type_t* type_)
    {
        long     pos = ftell(in_);
        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in_));
        if(format != format_t::dense_matrix)
        {
            std::cerr << " wrong format, not flagged as a dense matrix. " << std::endl;
            return status_t::invalid_format;
        }

        ROCSPARSEIO_CHECK(fread_scalar<size_t>(order_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(type_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

} // namespace rocsparseio

//
//
//
namespace rocsparseio
{
    inline status_t fwrite_sparse_csx(FILE*       out_,
                                      direction_t dir_,
                                      size_t      m_,
                                      size_t      n_,
                                      size_t      nnz_,
                                      type_t      ptr_type_,
                                      const void* __restrict__ ptr_,
                                      type_t ind_type_,
                                      const void* __restrict__ ind_,
                                      type_t data_type_,
                                      const void* __restrict__ data_,
                                      index_base_t base_)
    {
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(format_t::sparse_csx, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(dir_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(ptr_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(base_, out_));
        switch(dir_)
        {
        case direction_t::row:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), m_ + 1, ptr_));
            break;
        }
        case direction_t::column:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), n_ + 1, ptr_));
            break;
        }
        }

        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), nnz_, ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, data_type_.size(), nnz_, data_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_csx(FILE*         in_,
                                              direction_t*  dir_,
                                              size_t*       m_,
                                              size_t*       n_,
                                              size_t*       nnz_,
                                              type_t*       ptr_type,
                                              type_t*       ind_type,
                                              type_t*       data_type,
                                              index_base_t* base)
    {
        long     pos = ftell(in_);
        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(dir_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(ptr_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(ind_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(data_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(base, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_csx(FILE* in_,
                                     void* __restrict__ ptr_,
                                     void* __restrict__ ind_,
                                     void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(size_t) * 9, SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ptr_));
        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    inline status_t fwrite_sparse_coo(FILE*  out_,
                                      size_t m_,
                                      size_t n_,
                                      size_t nnz_,
                                      type_t row_ind_type_,
                                      const void* __restrict__ row_ind_,
                                      type_t col_ind_type_,
                                      const void* __restrict__ col_ind_,
                                      type_t data_type_,
                                      const void* __restrict__ data_,
                                      index_base_t base_)
    {
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(format_t::sparse_coo, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(row_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(col_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(base_, out_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, row_ind_type_.size(), nnz_, row_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, col_ind_type_.size(), nnz_, col_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, data_type_.size(), nnz_, data_));
        return status_t::success;
    };

    inline status_t fread_metadata_sparse_coo(FILE*         in_,
                                              size_t*       m_,
                                              size_t*       n_,
                                              size_t*       nnz_,
                                              type_t*       row_ind_type,
                                              type_t*       col_ind_type,
                                              type_t*       data_type,
                                              index_base_t* base)
    {
        long     pos = ftell(in_);
        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(row_ind_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(col_ind_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(data_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(base, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_coo(FILE* in_,
                                     void* __restrict__ row_ind_,
                                     void* __restrict__ col_ind_,
                                     void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(size_t) * 8, SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, row_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, col_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{

    inline status_t fwrite_sparse_gebsx(FILE*       out_,
                                        direction_t dir_,
                                        direction_t dirb_,
                                        size_t      mb_,
                                        size_t      nb_,
                                        size_t      nnzb_,
                                        size_t      row_block_dim_,
                                        size_t      col_block_dim_,
                                        type_t      ptr_type_,
                                        const void* __restrict__ ptr_,
                                        type_t ind_type_,
                                        const void* __restrict__ ind_,
                                        type_t data_type_,
                                        const void* __restrict__ data_,
                                        index_base_t base_)
    {

        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(format_t::sparse_gebsx, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(dir_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(dirb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(mb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(nb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(nnzb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(row_block_dim_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(col_block_dim_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(ptr_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<size_t>(base_, out_));

        switch(dir_)
        {
        case direction_t::row:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), mb_ + 1, ptr_));
            break;
        }
        case direction_t::column:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), nb_ + 1, ptr_));
            break;
        }
        }

        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), nnzb_, ind_));
        ROCSPARSEIO_CHECK(
            fwrite_array(out_, data_type_.size(), nnzb_ * row_block_dim_ * col_block_dim_, data_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_gebsx(FILE*         in_,
                                                direction_t*  dir_,
                                                direction_t*  dirb_,
                                                size_t*       mb_,
                                                size_t*       nb_,
                                                size_t*       nnzb_,
                                                size_t*       row_block_dim_,
                                                size_t*       col_block_dim_,
                                                type_t*       ptr_type,
                                                type_t*       ind_type,
                                                type_t*       data_type,
                                                index_base_t* base)
    {
        long     pos = ftell(in_);
        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(dir_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(dirb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(mb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(nnzb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(row_block_dim_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(col_block_dim_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(ptr_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(ind_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(data_type, in_));
        ROCSPARSEIO_CHECK(fread_scalar<size_t>(base, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_gebsx(FILE* in_,
                                       void* __restrict__ ptr_,
                                       void* __restrict__ ind_,
                                       void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(size_t) * 12, SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ptr_));
        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    template <>
    inline type_t type_t::convert<int32_t>()
    {
        return type_t::int32;
    };
    template <>
    inline type_t type_t::convert<int64_t>()
    {
        return type_t::int64;
    };

    template <>
    inline type_t type_t::convert<float>()
    {
        return type_t::float32;
    };
    template <>
    inline type_t type_t::convert<double>()
    {
        return type_t::float64;
    };

    template <typename... Ts>
    inline status_t read_format(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_format(handle->f, ts...);
    }

    inline status_t open(rocsparseio_handle* p_handle, rwmode_t mode, const char* filename, ...)
    {
        char filename_[512];
        {
            va_list args;
            va_start(args, filename);
            if(vsnprintf(filename_, 512, filename, args) >= 512)
            {
                std::cerr << "string is too long and is trucated. " << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }

        rocsparseio_handle h = new _rocsparseio_handle(mode, filename_);
        p_handle[0]          = h;

        h->f = nullptr;
        switch(mode)
        {
        case rwmode_t::read:
        {
            h->f = fopen(filename_, "rb");
            if(h->f == nullptr)
            {
                return status_t::invalid_file;
            }

            {
                size_t ref_value[2]{};
                char*  p = (char*)&ref_value[0];
                sprintf(p, "ROCSPARSEIO.%d", ROCSPARSEIO_VERSION_MAJOR);
                size_t value[2]{};
                if(2 != fread(&value[0], sizeof(size_t), 2, h->f))
                {
                    return status_t::invalid_file_operation;
                }
                //	    std::cout << "read " << value[0] << " " << value[1] <<
                //std::endl;
                if(ref_value[0] != value[0] || ref_value[1] != value[1])
                {
                    std::cerr << "incompatible rocsparseio version: " << std::endl;
                    std::cerr << "   expected      : " << ref_value[0] << "." << ref_value[1]
                              << std::endl;
                    std::cerr << "   from file     : " << value[0] << "." << value[1] << std::endl;
                    return status_t::invalid_file;
                }
            }

            break;
        }

        case rwmode_t::write:
        {

            h->f = fopen(filename_, "wb");
            if(h->f == nullptr)
            {
                return status_t::invalid_file;
            }

            //
            // WRITE HEADER
            //
            {
                size_t value[2]{};
                char*  p = (char*)&value;
                sprintf(p, "ROCSPARSEIO.%d", ROCSPARSEIO_VERSION_MAJOR);
                if(2 != fwrite(&value[0], sizeof(size_t), 2, h->f))
                {
                    return status_t::invalid_file_operation;
                }
            }

            break;
        }
        }
        return status_t::success;
    }

    //!
    //! @brief Close.
    //!
    inline status_t close(rocsparseio_handle handle)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        if(handle->f != nullptr)
        {
            fclose(handle->f);
        }
        delete handle;
        return status_t::success;
    }

    template <typename... Ts>
    inline status_t write_dense_vector(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fwrite_dense_vector(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_dense_vector(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_dense_vector(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_metadata_dense_vector(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_metadata_dense_vector(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t write_dense_matrix(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fwrite_dense_matrix(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_metadata_dense_matrix(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_metadata_dense_matrix(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_dense_matrix(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_dense_matrix(handle->f, ts...);
    }

    template <typename T, typename J>
    inline status_t write_dense_matrix_template(
        rocsparseio_handle handle, order_t order_, J m_, J n_, const T* data_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        ROCSPARSEIO_CHECK_ARG(handle->mode != rwmode_t::write, status_t::invalid_mode);
        return fwrite_dense_matrix(handle->f,
                                   order_,
                                   static_cast<size_t>(m_),
                                   static_cast<size_t>(n_),
                                   type_t::convert<T>(),
                                   data_,
                                   (order_ == ROCSPARSEIO_ORDER_ROW) ? static_cast<size_t>(n_)
                                                                     : static_cast<size_t>(m_));
    };

    template <typename... Ts>
    inline status_t write_sparse_csx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fwrite_sparse_csx(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_csx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_metadata_sparse_csx(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_sparse_csx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_sparse_csx(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t write_sparse_coo(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fwrite_sparse_coo(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_coo(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_metadata_sparse_coo(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_sparse_coo(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_sparse_coo(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t write_sparse_gebsx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fwrite_sparse_gebsx(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_gebsx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_metadata_sparse_gebsx(handle->f, ts...);
    }

    template <typename... Ts>
    inline status_t read_sparse_gebsx(rocsparseio_handle handle, Ts&&... ts)
    {
        ROCSPARSEIO_CHECK_ARG(!handle, status_t::invalid_handle);
        return fread_sparse_gebsx(handle->f, ts...);
    }

} // namespace rocsparseio
