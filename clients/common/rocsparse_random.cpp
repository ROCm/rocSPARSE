/*! \file */
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

#include "rocsparse_random.hpp"

#include <iostream>

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
rocsparse_rng_t rocsparse_rng(69069);
rocsparse_rng_t rocsparse_rng_nan(69069);
rocsparse_rng_t rocsparse_seed(rocsparse_rng);

void rocsparse_rng_set(rocsparse_rng_t a)
{
    rocsparse_rng = a;
}

void rocsparse_seed_set(rocsparse_rng_t a)
{
    rocsparse_seed = a;
}

void rocsparse_rng_nan_set(rocsparse_rng_t a)
{
    rocsparse_rng_nan = a;
}

rocsparse_rng_t& rocsparse_rng_get()
{
    return rocsparse_rng;
}

rocsparse_rng_t& rocsparse_seed_get()
{
    return rocsparse_seed;
}

rocsparse_rng_t& rocsparse_rng_nan_get()
{
    return rocsparse_rng_nan;
}

#define RANDOM_CACHE_SIZE 1024

int rocsparse_rand_uniform_float_idx;
int rocsparse_rand_uniform_double_idx;
int rocsparse_rand_normal_double_idx;

static int s_rand_uniform_float_init  = 0;
static int s_rand_uniform_double_init = 0;
static int s_rand_normal_double_init  = 0;

// float random uniform numbers between 0.0f - 1.0f
static float s_rand_uniform_float_array[RANDOM_CACHE_SIZE];
// double random uniform numbers between 0.0 - 1.0
static double s_rand_uniform_double_array[RANDOM_CACHE_SIZE];
// double random normal numbers between 0.0 - 1.0
static double s_rand_normal_double_array[RANDOM_CACHE_SIZE];

float rocsparse_uniform_float(float a, float b)
{
    if(!s_rand_uniform_float_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_uniform_float_array[i]
                = std::uniform_real_distribution<float>(0.0f, 1.0f)(rocsparse_rng_get());
        }
        s_rand_uniform_float_init = 1;
    }

    rocsparse_rand_uniform_float_idx
        = (rocsparse_rand_uniform_float_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + s_rand_uniform_float_array[rocsparse_rand_uniform_float_idx] * (b - a);
}

double rocsparse_uniform_double(double a, double b)
{
    if(!s_rand_uniform_double_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_uniform_double_array[i]
                = std::uniform_real_distribution<double>(0.0, 1.0)(rocsparse_rng_get());
        }
        s_rand_uniform_double_init = 1;
    }

    rocsparse_rand_uniform_double_idx
        = (rocsparse_rand_uniform_double_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + s_rand_uniform_double_array[rocsparse_rand_uniform_double_idx] * (b - a);
}

int rocsparse_uniform_int(int a, int b)
{
    return rocsparse_uniform_float(static_cast<float>(a), static_cast<float>(b));
}

double rocsparse_normal_double()
{
    if(!s_rand_normal_double_init)
    {
        for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
        {
            s_rand_normal_double_array[i]
                = std::normal_distribution<double>(0.0, 1.0)(rocsparse_rng_get());
        }
        s_rand_normal_double_init = 1;
    }

    rocsparse_rand_normal_double_idx
        = (rocsparse_rand_normal_double_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return s_rand_normal_double_array[rocsparse_rand_normal_double_idx];
}
