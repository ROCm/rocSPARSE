/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdio.h>
#include <rocsparse.h>

int main(int argc, char *argv[])
{
    rocsparseHandle_t handle;
    rocsparseCreate(&handle);

    int version;
    rocsparseGetVersion(handle, &version);

    printf("rocSPARSE version %d.%d.%d\n",
           version / 100000,
           version / 100 % 1000,
           version % 100);

    rocsparseDestroy(handle);

    return 0;
}
