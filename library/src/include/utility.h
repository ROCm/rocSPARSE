/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#include "rocsparse.h"
#include "handle.h"
#include "logging.h"

#include <fstream>
#include <string>
#include <algorithm>

// if trace logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(rocsparse_handle handle, H head, Ts&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_trace)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;
            log_arguments(*os, comma_separator, head, xs...);
        }
    }
}

// if bench logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocsparse-bench.
template <typename H, typename... Ts>
void log_bench(rocsparse_handle handle, H head, std::string precision, Ts&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_bench)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_bench_os;
            log_arguments(*os, space_separator, head, precision, xs...);
        }
    }
}

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(std::string input_string)
{
    if(std::is_same<T, float>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 's');
    }
    else if(std::is_same<T, double>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'd');
    }
/*
    else if(std::is_same<T, rocsparse_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocsparse_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocsparse_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
*/
    return input_string;
}

#endif // UTILITY_H
