!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

! [doc example start]
program example_fortran_gebsr2csr
    use iso_c_binding
    use rocsparse

    implicit none

    interface
        function hipMalloc(ptr, size) bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            integer :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            integer :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            integer :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    integer, parameter :: hipMemcpyHostToDevice = 1
    integer, parameter :: hipMemcpyDeviceToHost = 2

    ! Matrix dimensions
    integer(c_int) :: mb, nb, row_block_dim, col_block_dim, m, n, nnzb
    
    ! Host arrays
    integer, dimension(3), target :: h_bsr_row_ptr
    integer, dimension(3), target :: h_bsr_col_ind
    real(c_float), dimension(18), target :: h_bsr_val
    
    ! Device pointers
    type(c_ptr) :: d_bsr_row_ptr, d_bsr_col_ind, d_bsr_val
    type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, bsr_descr, csr_descr

    ! Initialize dimensions
    mb = 2
    nb = 2
    row_block_dim = 2
    col_block_dim = 3
    m = mb * row_block_dim
    n = nb * col_block_dim
    
    ! Initialize host data
    h_bsr_row_ptr = (/0, 1, 3/)
    h_bsr_col_ind = (/0, 0, 1/)
    h_bsr_val = (/1.0, 0.0, 4.0, 2.0, 0.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 9.0, &
                  7.0, 0.0, 8.0, 6.0, 0.0, 0.0/)
    
    nnzb = h_bsr_row_ptr(mb + 1) - h_bsr_row_ptr(1)

    ! Allocate device memory for BSR matrix
    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, int(mb + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(nnzb * row_block_dim * col_block_dim, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(d_bsr_row_ptr, c_loc(h_bsr_row_ptr), int(mb + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_bsr_col_ind, c_loc(h_bsr_col_ind), int(nnzb, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_bsr_val, c_loc(h_bsr_val), &
                             int(nnzb * row_block_dim * col_block_dim, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create CSR arrays on device
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, &
                             int(nnzb * row_block_dim * col_block_dim, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnzb * row_block_dim * col_block_dim, c_size_t) * 4))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(bsr_descr))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero))
    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero))

    ! Format conversion
    call ROCSPARSE_CHECK(rocsparse_sgebsr2csr(handle, rocsparse_direction_column, mb, nb, &
                                               bsr_descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind, &
                                               row_block_dim, col_block_dim, csr_descr, d_csr_val, &
                                               d_csr_row_ptr, d_csr_col_ind))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: h_csr_row_ptr(:)
        integer :: i

        allocate(h_csr_row_ptr(m + 1))
        call HIP_CHECK(hipMemcpy(c_loc(h_csr_row_ptr), d_csr_row_ptr, &
            int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A)',advance='no') 'CSR row_ptr: '
        do i = 1, m + 1
            write(*,fmt='(I0,A)',advance='no') h_csr_row_ptr(i), ' '
        end do
        write(*,*)

        deallocate(h_csr_row_ptr)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_csr_val))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_gebsr2csr
! [doc example end]
