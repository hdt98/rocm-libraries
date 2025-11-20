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
program example_fortran_csr2bsr
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
    integer(c_int) :: m, n, block_dim, nnz, mb, nb
    integer(c_int), target :: nnzb
    
    ! Host arrays
    integer, dimension(5), target :: h_csr_row_ptr
    integer, dimension(9), target :: h_csr_col_ind
    real(c_float), dimension(9), target :: h_csr_val
    
    ! Device pointers
    type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
    type(c_ptr) :: d_bsr_row_ptr, d_bsr_col_ind, d_bsr_val
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, csr_descr, bsr_descr

    ! Initialize dimensions
    m = 4
    n = 6
    block_dim = 2
    nnz = 9
    mb = (m + block_dim - 1) / block_dim
    nb = (n + block_dim - 1) / block_dim
    
    ! Initialize host data
    h_csr_row_ptr = (/0, 2, 4, 7, 9/)
    h_csr_col_ind = (/0, 1, 1, 2, 0, 3, 4, 2, 4/)
    h_csr_val = (/1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 6.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Allocate device memory for CSR format
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(bsr_descr))

    ! Allocate BSR row pointer array
    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, int(mb + 1, c_size_t) * 4))

    ! Compute the number of non-zero block entries
    call ROCSPARSE_CHECK(rocsparse_csr2bsr_nnz(handle, rocsparse_direction_row, m, n, &
                                                csr_descr, d_csr_row_ptr, d_csr_col_ind, &
                                                block_dim, bsr_descr, d_bsr_row_ptr, c_loc(nnzb)))

    ! Allocate BSR column indices and values arrays
    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(block_dim * block_dim * nnzb, c_size_t) * 4))

    ! Convert CSR to BSR
    call ROCSPARSE_CHECK(rocsparse_scsr2bsr(handle, rocsparse_direction_row, m, n, csr_descr, &
                                            d_csr_val, d_csr_row_ptr, d_csr_col_ind, block_dim, &
                                            bsr_descr, d_bsr_val, d_bsr_row_ptr, d_bsr_col_ind))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: h_bsr_row_ptr(:)
        integer :: i

        allocate(h_bsr_row_ptr(mb + 1))
        call HIP_CHECK(hipMemcpy(c_loc(h_bsr_row_ptr), d_bsr_row_ptr, int(mb + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'nnzb (number of non-zero blocks): ', nnzb
        write(*,fmt='(A)',advance='no') 'BSR row_ptr: '
        do i = 1, mb + 1
            write(*,fmt='(I0,A)',advance='no') h_bsr_row_ptr(i), ' '
        end do
        write(*,*)

        deallocate(h_bsr_row_ptr)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2bsr
! [doc example end]
