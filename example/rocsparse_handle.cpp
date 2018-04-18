/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <rocsparse.h>

int main(int argc, char *argv[])
{
    rocsparseHandle_t handle;
    rocsparseCreate(&handle);

    rocsparseDestroy(handle);

    return 0;
}
