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
program example_fortran_csr2gebsr
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
    integer(c_int) :: m, n, row_block_dim, col_block_dim, nnz, mb, nb
    integer(c_int), target :: nnzb
    integer(c_int) :: dir
    
    ! Host arrays
    integer, dimension(5), target :: hcsr_row_ptr
    integer, dimension(9), target :: hcsr_col_ind
    real(c_float), dimension(9), target :: hcsr_val
    
    ! Device pointers
    type(c_ptr) :: dcsr_row_ptr, dcsr_col_ind, dcsr_val
    type(c_ptr) :: dbsr_row_ptr, dbsr_col_ind, dbsr_val
    type(c_ptr) :: buffer
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, csr_descr, bsr_descr
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m = 4
    n = 6
    row_block_dim = 2
    col_block_dim = 3
    nnz = 9
    mb = (m + row_block_dim - 1) / row_block_dim
    nb = (n + col_block_dim - 1) / col_block_dim
    dir = rocsparse_direction_row
    
    ! Initialize host data
    hcsr_row_ptr = (/0, 2, 4, 7, 9/)
    hcsr_col_ind = (/0, 1, 1, 2, 0, 3, 4, 2, 4/)
    hcsr_val = (/1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 6.0/)

    ! Allocate device memory for CSR format
    call HIP_CHECK(hipMalloc(dcsr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val, int(nnz, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dcsr_row_ptr, c_loc(hcsr_row_ptr), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_col_ind, c_loc(hcsr_col_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_val, c_loc(hcsr_val), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(bsr_descr))

    ! Get buffer size
    call ROCSPARSE_CHECK(rocsparse_scsr2gebsr_buffer_size(handle, dir, m, n, csr_descr, &
                                                           dcsr_val, dcsr_row_ptr, dcsr_col_ind, &
                                                           row_block_dim, col_block_dim, &
                                                           c_loc(buffer_size)))
    call HIP_CHECK(hipMalloc(buffer, buffer_size))

    ! Allocate BSR row pointer array
    call HIP_CHECK(hipMalloc(dbsr_row_ptr, int(mb + 1, c_size_t) * 4))

    ! Compute nnzb
    call ROCSPARSE_CHECK(rocsparse_csr2gebsr_nnz(handle, dir, m, n, csr_descr, dcsr_row_ptr, &
                                                  dcsr_col_ind, bsr_descr, dbsr_row_ptr, &
                                                  row_block_dim, col_block_dim, c_loc(nnzb), buffer))

    ! Allocate GEBSR column indices and values
    call HIP_CHECK(hipMalloc(dbsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val, int(nnzb * row_block_dim * col_block_dim, c_size_t) * 4))

    ! Convert CSR to GEBSR
    call ROCSPARSE_CHECK(rocsparse_scsr2gebsr(handle, dir, m, n, csr_descr, dcsr_val, &
                                               dcsr_row_ptr, dcsr_col_ind, bsr_descr, dbsr_val, &
                                               dbsr_row_ptr, dbsr_col_ind, row_block_dim, &
                                               col_block_dim, buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: h_bsr_row_ptr(:)
        integer :: i

        allocate(h_bsr_row_ptr(mb + 1))
        call HIP_CHECK(hipMemcpy(c_loc(h_bsr_row_ptr), dbsr_row_ptr, &
            int(mb + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'nnzb: ', nnzb
        write(*,fmt='(A)',advance='no') 'GEBSR row_ptr: '
        do i = 1, mb + 1
            write(*,fmt='(I0,A)',advance='no') h_bsr_row_ptr(i), ' '
        end do
        write(*,*)

        deallocate(h_bsr_row_ptr)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(buffer))
    call HIP_CHECK(hipFree(dcsr_row_ptr))
    call HIP_CHECK(hipFree(dcsr_col_ind))
    call HIP_CHECK(hipFree(dcsr_val))
    call HIP_CHECK(hipFree(dbsr_row_ptr))
    call HIP_CHECK(hipFree(dbsr_col_ind))
    call HIP_CHECK(hipFree(dbsr_val))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2gebsr
! [doc example end]
