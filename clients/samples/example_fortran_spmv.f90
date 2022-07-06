!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if

end subroutine HIP_CHECK

subroutine ROCSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocsparse error'
        stop
    end if

end subroutine ROCSPARSE_CHECK

program example_fortran_spmv
    use iso_c_binding
    use rocsparse

    implicit none

    interface
        function hipMalloc(ptr, size) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            integer :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            integer :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            integer :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy

        function hipMemset(dst, val, size) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            integer :: hipMemset
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset

        function hipDeviceSynchronize() &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
            integer :: hipDeviceSynchronize
        end function hipDeviceSynchronize

        function hipDeviceReset() &
                bind(c, name = 'hipDeviceReset')
            use iso_c_binding
            implicit none
            integer :: hipDeviceReset
        end function hipDeviceReset
    end interface

    integer, dimension(:), allocatable, target :: h_csr_row_ptr, h_csr_col_ind
    real(8), dimension(:), allocatable, target :: h_csr_val, h_x, h_y, h_y_gold

    type(c_ptr) :: d_csr_row_ptr
    type(c_ptr) :: d_csr_col_ind
    type(c_ptr) :: d_csr_val
    type(c_ptr) :: d_x
    type(c_ptr) :: d_y
    type(c_ptr) :: d_coo_row_ind
    type(c_ptr) :: d_ell_col_ind
    type(c_ptr) :: d_ell_val
    type(c_ptr) :: d_bsr_row_ptr
    type(c_ptr) :: d_bsr_col_ind
    type(c_ptr) :: d_bsr_val

    integer(c_int) :: M, N, nnz
    integer(c_int) :: dim_x, dim_y
    integer(c_int) :: row, col
    integer(c_int) :: ix, iy, sx, sy

    real(c_double), target :: alpha
    real(c_double), target :: beta

    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info
    type(c_ptr) :: hyb
    type(c_ptr) :: d_nnzb
    type(c_ptr) :: d_alpha
    type(c_ptr) :: d_beta

    integer :: Mb, Nb, bsr_dim
    integer, target :: nnzb, ell_width
    integer :: version

    character(len=12) :: rev

    integer i
    integer tbegin(8)
    integer tend(8)
    real(8) timing
    real(8) gflops
    real(8) gbyte
    real(8) acc

!   Sample Laplacian on 2D domain
    dim_x = 3000
    dim_y = 3000

!   Dimensions
    M = dim_x * dim_y
    N = dim_x * dim_y

!   Allocate CSR arrays and vectors
    allocate(h_csr_row_ptr(M + 1), h_csr_col_ind(9 * M), h_csr_val(9 * M))
    allocate(h_x(N))
    allocate(h_y_gold(M))
    allocate(h_y(M))

!   Initialize with 0 index base
    h_csr_row_ptr(1) = 0

    nnz = 0

!   Fill host arrays
    do iy = 0, dim_y - 1
        do ix = 0, dim_x - 1
            row = iy * dim_x + ix
            do sy = -1, 1
                if(iy + sy .gt. -1 .and. iy + sy .lt. dim_y) then
                    do sx = -1, 1
                        if(ix + sx .gt. -1 .and. ix + sx .lt. dim_x) then
                            col = row + sy * dim_x + sx
                            h_csr_col_ind(nnz + 1) = col
                            if(col .eq. row) then
                                h_csr_val(nnz + 1) = 8
                            else
                                h_csr_val(nnz + 1) = -1
                            endif
                            nnz = nnz + 1
                        end if
                    end do
                end if
            end do
            h_csr_row_ptr(row + 2) = nnz
        end do
    end do

!   Initialize x and y
    h_x(1:N) = 1

!   Scalars
    alpha = 1
    beta  = 0

!   Print assembled matrix sizes
    write(*,fmt='(A,I0,A,I0,A,I0,A)') "2D Laplacian matrix: ", M, " x ", N, " with ", nnz, " non-zeros"

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, (int(M, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_x, int(N, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_y, int(M, c_size_t) * 8))

!   Set y to zero
    call HIP_CHECK(hipMemset(d_y, 0, int(M, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), (int(M, c_size_t) + 1) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_x, c_loc(h_x), int(N, c_size_t) * 8, 1))

!   Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

!   Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))
    call ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev))

!   Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0,A,A)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100), '-', rev

!   Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr));

!   Create matrix info structure
    call ROCSPARSE_CHECK(rocsparse_create_mat_info(info));

!   Collect meta data running analysis step
    call ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle, &
                                                   rocsparse_operation_none, &
                                                   M, &
                                                   N, &
                                                   nnz, &
                                                   descr, &
                                                   d_csr_val, &
                                                   d_csr_row_ptr, &
                                                   d_csr_col_ind, &
                                                   info))

!   Warm up
    call ROCSPARSE_CHECK(rocsparse_dcsrmv(handle, &
                                          rocsparse_operation_none, &
                                          M, &
                                          N, &
                                          nnz, &
                                          c_loc(alpha), &
                                          descr, &
                                          d_csr_val, &
                                          d_csr_row_ptr, &
                                          d_csr_col_ind, &
                                          info, &
                                          d_x, &
                                          c_loc(beta), &
                                          d_y))

!   Start time measurement
    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tbegin)

    do i = 1, 200
        call ROCSPARSE_CHECK(rocsparse_dcsrmv(handle, &
                                              rocsparse_operation_none, &
                                              M, &
                                              N, &
                                              nnz, &
                                              c_loc(alpha), &
                                              descr, &
                                              d_csr_val, &
                                              d_csr_row_ptr, &
                                              d_csr_col_ind, &
                                              info, &
                                              d_x, &
                                              c_loc(beta), &
                                              d_y))
    end do

    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tend)
    tbegin = tend - tbegin;
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    gbyte  = ((M + N + nnz) * 8d0 + (M + 1 + nnz) * 4d0) / timing / 1000000d0
    gflops = (2d0 * nnz) / timing / 1000000d0
    write(*,fmt='(A,F0.2,A,F0.2,A,F0.2,A)') '[rocsparse_dcsrmv] took ', &
        timing, ' msec; ', gbyte, ' GB/s; ', gflops, ' GFlop/s'

!   Verify CSR result
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(M, c_size_t) * 8, 2))

    do row = 1, M
        acc = beta * h_y_gold(row)
        do i = h_csr_row_ptr(row) + 1, h_csr_row_ptr(row + 1)
            col = h_csr_col_ind(i) + 1
            acc = acc + h_csr_val(i) * h_x(col)
        end do
        h_y_gold(row) = alpha * acc

        if(h_y_gold(row) .ne. h_y(row)) then
            write(*,*) '[rocsparse_dcsrmv] ERROR: ', h_y_gold(row), '!=', h_y(row)
        end if
    end do

!   Convert to COO
    call HIP_CHECK(hipMalloc(d_coo_row_ind, int(nnz, c_size_t) * 4))

    call ROCSPARSE_CHECK(rocsparse_csr2coo(handle, &
                                           d_csr_row_ptr, &
                                           nnz, &
                                           M, &
                                           d_coo_row_ind, &
                                           rocsparse_index_base_zero))

!   Warm up
    call ROCSPARSE_CHECK(rocsparse_dcoomv(handle, &
                                          rocsparse_operation_none, &
                                          M, &
                                          N, &
                                          nnz, &
                                          c_loc(alpha), &
                                          descr, &
                                          d_csr_val, &
                                          d_coo_row_ind, &
                                          d_csr_col_ind, &
                                          d_x, &
                                          c_loc(beta), &
                                          d_y))

!   Start time measurement
    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tbegin)

    do i = 1, 200
        call ROCSPARSE_CHECK(rocsparse_dcoomv(handle, &
                                              rocsparse_operation_none, &
                                              M, &
                                              N, &
                                              nnz, &
                                              c_loc(alpha), &
                                              descr, &
                                              d_csr_val, &
                                              d_coo_row_ind, &
                                              d_csr_col_ind, &
                                              d_x, &
                                              c_loc(beta), &
                                              d_y))
    end do

    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tend)
    tbegin = tend - tbegin;
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    gbyte  = ((M + N + nnz) * 8d0 + (M + 1 + nnz) * 4d0) / timing / 1000000d0
    gflops = (2d0 * nnz) / timing / 1000000d0
    write(*,fmt='(A,F0.2,A,F0.2,A,F0.2,A)') '[rocsparse_dcoomv] took ', &
        timing, ' msec; ', gbyte, ' GB/s; ', gflops, ' GFlop/s'

!   Verify COO result
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(M, c_size_t) * 8, 2))

    do row = 1, M
        if(h_y_gold(row) .ne. h_y(row)) then
            write(*,*) '[rocsparse_dcoomv] ERROR: ', h_y_gold(row), '!=', h_y(row)
        end if
    end do

!   Free COO structures
    call HIP_CHECK(hipFree(d_coo_row_ind))

!   Convert to ELL
    call HIP_CHECK(hipMalloc(d_ell_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_ell_val, int(nnz, c_size_t) * 8))

    call ROCSPARSE_CHECK(rocsparse_csr2ell_width(handle, &
                                                 M, &
                                                 descr, &
                                                 d_csr_row_ptr, &
                                                 descr, &
                                                 c_loc(ell_width)))

    call ROCSPARSE_CHECK(rocsparse_dcsr2ell(handle, &
                                            M, &
                                            descr, &
                                            d_csr_val, &
                                            d_csr_row_ptr, &
                                            d_csr_col_ind, &
                                            descr, &
                                            ell_width, &
                                            d_ell_val, &
                                            d_ell_col_ind))

!   Warm up
    call ROCSPARSE_CHECK(rocsparse_dellmv(handle, &
                                          rocsparse_operation_none, &
                                          M, &
                                          N, &
                                          c_loc(alpha), &
                                          descr, &
                                          d_ell_val, &
                                          d_ell_col_ind, &
                                          ell_width, &
                                          d_x, &
                                          c_loc(beta), &
                                          d_y))

!   Start time measurement
    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tbegin)

    do i = 1, 200
        call ROCSPARSE_CHECK(rocsparse_dellmv(handle, &
                                              rocsparse_operation_none, &
                                              M, &
                                              N, &
                                              c_loc(alpha), &
                                              descr, &
                                              d_ell_val, &
                                              d_ell_col_ind, &
                                              ell_width, &
                                              d_x, &
                                              c_loc(beta), &
                                              d_y))
    end do

    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tend)
    tbegin = tend - tbegin;
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    gbyte  = ((M + N + nnz) * 8d0 + (M + 1 + nnz) * 4d0) / timing / 1000000d0
    gflops = (2d0 * nnz) / timing / 1000000d0
    write(*,fmt='(A,F0.2,A,F0.2,A,F0.2,A)') '[rocsparse_dellmv] took ', &
        timing, ' msec; ', gbyte, ' GB/s; ', gflops, ' GFlop/s'

!   Verify ELL result
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(M, c_size_t) * 8, 2))

    do row = 1, M
        if(h_y_gold(row) .ne. h_y(row)) then
            write(*,*) '[rocsparse_dellmv] ERROR: ', h_y_gold(row), '!=', h_y(row)
        end if
    end do

!   Free ELL structures
    call HIP_CHECK(hipFree(d_ell_col_ind))
    call HIP_CHECK(hipFree(d_ell_val))

!   Convert to HYB
    call ROCSPARSE_CHECK(rocsparse_create_hyb_mat(hyb))

    call ROCSPARSE_CHECK(rocsparse_dcsr2hyb(handle, &
                                            M, &
                                            N, &
                                            descr, &
                                            d_csr_val, &
                                            d_csr_row_ptr, &
                                            d_csr_col_ind, &
                                            hyb, &
                                            0, &
                                            rocsparse_hyb_partition_auto))

!   Warm up
    call ROCSPARSE_CHECK(rocsparse_dhybmv(handle, &
                                          rocsparse_operation_none, &
                                          c_loc(alpha), &
                                          descr, &
                                          hyb, &
                                          d_x, &
                                          c_loc(beta), &
                                          d_y))

!   Start time measurement
    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tbegin)

    do i = 1, 200
        call ROCSPARSE_CHECK(rocsparse_dhybmv(handle, &
                                              rocsparse_operation_none, &
                                              c_loc(alpha), &
                                              descr, &
                                              hyb, &
                                              d_x, &
                                              c_loc(beta), &
                                              d_y))
    end do

    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tend)
    tbegin = tend - tbegin;
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    gbyte  = ((M + N + nnz) * 8d0 + (M + 1 + nnz) * 4d0) / timing / 1000000d0
    gflops = (2d0 * nnz) / timing / 1000000d0
    write(*,fmt='(A,F0.2,A,F0.2,A,F0.2,A)') '[rocsparse_dhybmv] took ', &
        timing, ' msec; ', gbyte, ' GB/s; ', gflops, ' GFlop/s'

!   Verify HYB result
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(M, c_size_t) * 8, 2))

    do row = 1, M
        if(h_y_gold(row) .ne. h_y(row)) then
            write(*,*) '[rocsparse_dhybmv] ERROR: ', h_y_gold(row), '!=', h_y(row)
        end if
    end do

!   Free HYB structures
    call ROCSPARSE_CHECK(rocsparse_destroy_hyb_mat(hyb))

!   Convert to BSR
    bsr_dim = 2
    Mb = (M + bsr_dim - 1) / bsr_dim
    Nb = (N + bsr_dim - 1) / bsr_dim

    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, (int(Mb, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_nnzb, int(4, c_size_t)))
    call HIP_CHECK(hipMalloc(d_alpha, int(8, c_size_t)))
    call HIP_CHECK(hipMalloc(d_beta, int(8, c_size_t)))

    call HIP_CHECK(hipMemcpy(d_alpha, c_loc(alpha), int(8, c_size_t), 1))
    call HIP_CHECK(hipMemcpy(d_beta, c_loc(beta), int(8, c_size_t), 1))

!   Test device pointer mode
    call ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device))
    call ROCSPARSE_CHECK(rocsparse_csr2bsr_nnz(handle, &
                                               rocsparse_direction_column, &
                                               M, &
                                               N, &
                                               descr, &
                                               d_csr_row_ptr, &
                                               d_csr_col_ind, &
                                               bsr_dim, &
                                               descr, &
                                               d_bsr_row_ptr, &
                                               d_nnzb))

!   Copy device nnzb to host
    call HIP_CHECK(hipMemcpy(c_loc(nnzb), d_nnzb, int(4, c_size_t), 2))

    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(nnzb, c_size_t) * bsr_dim * bsr_dim * 8))

    call ROCSPARSE_CHECK(rocsparse_dcsr2bsr(handle, &
                                            rocsparse_direction_column, &
                                            M, &
                                            N, &
                                            descr, &
                                            d_csr_val, &
                                            d_csr_row_ptr, &
                                            d_csr_col_ind, &
                                            bsr_dim, &
                                            descr, &
                                            d_bsr_val, &
                                            d_bsr_row_ptr, &
                                            d_bsr_col_ind))

!   Warm up
    call ROCSPARSE_CHECK(rocsparse_dbsrmv(handle, &
                                          rocsparse_direction_column, &
                                          rocsparse_operation_none, &
                                          Mb, &
                                          Nb, &
                                          nnzb, &
                                          d_alpha, &
                                          descr, &
                                          d_bsr_val, &
                                          d_bsr_row_ptr, &
                                          d_bsr_col_ind, &
                                          bsr_dim, &
                                          d_x, &
                                          d_beta, &
                                          d_y))

!   Start time measurement
    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tbegin)

    do i = 1, 200
        call ROCSPARSE_CHECK(rocsparse_dbsrmv(handle, &
                                              rocsparse_direction_column, &
                                              rocsparse_operation_none, &
                                              Mb, &
                                              Nb, &
                                              nnzb, &
                                              d_alpha, &
                                              descr, &
                                              d_bsr_val, &
                                              d_bsr_row_ptr, &
                                              d_bsr_col_ind, &
                                              bsr_dim, &
                                              d_x, &
                                              d_beta, &
                                              d_y))
    end do

    call HIP_CHECK(hipDeviceSynchronize())
    call date_and_time(values = tend)
    tbegin = tend - tbegin;
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    gbyte  = ((M + N + nnz) * 8d0 + (M + 1 + nnz) * 4d0) / timing / 1000000d0
    gflops = (2d0 * nnz) / timing / 1000000d0
    write(*,fmt='(A,F0.2,A,F0.2,A,F0.2,A)') '[rocsparse_dbsrmv] took ', &
        timing, ' msec; ', gbyte, ' GB/s; ', gflops, ' GFlop/s'

!   Verify BSR result
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(M, c_size_t) * 8, 2))

    do row = 1, M
        if(h_y_gold(row) .ne. h_y(row)) then
            write(*,*) '[rocsparse_dbsrmv] ERROR: ', h_y_gold(row), '!=', h_y(row)
        end if
    end do

!   Free BSR structures
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))

!   Free host memory
    deallocate(h_csr_row_ptr, h_csr_col_ind, h_csr_val)
    deallocate(h_x, h_y)

!   Free device memory
    call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_x))
    call HIP_CHECK(hipFree(d_y))

!   Free rocSPARSE structures
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

    call HIP_CHECK(hipDeviceReset())

end program example_fortran_spmv
