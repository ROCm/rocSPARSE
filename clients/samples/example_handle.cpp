/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdio.h>
#include <rocsparse.h>

int main(int argc, char* argv[])
{
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    int version;
    rocsparse_get_version(handle, &version);

    printf("rocSPARSE version %d.%d.%d\n", version / 100000, version / 100 % 1000, version % 100);

    rocsparse_destroy_handle(handle);

    return 0;
}
