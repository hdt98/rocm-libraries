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
program example_fortran_bsr2csr
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
    end interface

    integer, parameter :: hipMemcpyHostToDevice = 1
    integer, parameter :: hipMemcpyDeviceToHost = 2

    ! Matrix dimensions
    integer(c_int) :: mb, nb, block_dim, m, n, nnzb, nnz
    
    ! Host arrays
    integer, dimension(3), target :: hbsr_row_ptr
    integer, dimension(5), target :: hbsr_col_ind
    real(c_float), dimension(20), target :: hbsr_val
    integer, allocatable, target :: hcsr_row_ptr(:), hcsr_col_ind(:)
    real(c_float), allocatable, target :: hcsr_val(:)
    
    ! Device pointers
    type(c_ptr) :: dbsr_row_ptr, dbsr_col_ind, dbsr_val
    type(c_ptr) :: dcsr_row_ptr, dcsr_col_ind, dcsr_val
    
    ! rocSPARSE handles
    type(c_ptr) :: handle
    type(c_ptr) :: bsr_descr, csr_descr
    
    ! Loop variables
    integer :: i, j, start_idx, end_idx
    real(c_float), allocatable :: temp(:)
    
    ! Initialize dimensions
    mb = 2
    nb = 3
    block_dim = 2
    m = mb * block_dim
    n = nb * block_dim
    nnzb = 5
    nnz = nnzb * block_dim * block_dim
    
    ! Initialize BSR matrix data
    !     1 4 2 1 0 0
    ! A = 0 2 3 5 0 0
    !     5 2 2 7 8 6
    !     9 3 9 1 6 1
    hbsr_row_ptr = (/0, 2, 5/)
    hbsr_col_ind = (/0, 1, 0, 1, 2/)
    hbsr_val = (/1.0, 0.0, 4.0, 2.0, 2.0, 3.0, 1.0, 5.0, 5.0, 9.0, &
                 2.0, 3.0, 2.0, 9.0, 7.0, 1.0, 8.0, 6.0, 6.0, 1.0/)
    
    ! Allocate device memory for BSR matrix
    call HIP_CHECK(hipMalloc(dbsr_row_ptr, int(mb + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val, int(nnzb * block_dim * block_dim, c_size_t) * 4))
    
    ! Copy BSR matrix to device
    call HIP_CHECK(hipMemcpy(dbsr_row_ptr, c_loc(hbsr_row_ptr), int(mb + 1, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_col_ind, c_loc(hbsr_col_ind), int(nnzb, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_val, c_loc(hbsr_val), int(nnzb * block_dim * block_dim, c_size_t) * 4, hipMemcpyHostToDevice))
    
    ! Allocate device memory for CSR matrix
    call HIP_CHECK(hipMalloc(dcsr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val, int(nnz, c_size_t) * 4))
    
    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))
    
    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(bsr_descr))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(csr_descr))
    
    ! Set matrix index base
    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero))
    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero))
    
    ! Perform BSR to CSR conversion
    call ROCSPARSE_CHECK(rocsparse_sbsr2csr(handle, &
                                            rocsparse_direction_column, &
                                            mb, &
                                            nb, &
                                            bsr_descr, &
                                            dbsr_val, &
                                            dbsr_row_ptr, &
                                            dbsr_col_ind, &
                                            block_dim, &
                                            csr_descr, &
                                            dcsr_val, &
                                            dcsr_row_ptr, &
                                            dcsr_col_ind))
    
    ! Allocate host memory for CSR result
    allocate(hcsr_row_ptr(m + 1))
    allocate(hcsr_col_ind(nnz))
    allocate(hcsr_val(nnz))
    allocate(temp(n))
    
    ! Copy result back to host
    call HIP_CHECK(hipMemcpy(c_loc(hcsr_row_ptr), dcsr_row_ptr, int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))
    call HIP_CHECK(hipMemcpy(c_loc(hcsr_col_ind), dcsr_col_ind, int(nnz, c_size_t) * 4, hipMemcpyDeviceToHost))
    call HIP_CHECK(hipMemcpy(c_loc(hcsr_val), dcsr_val, int(nnz, c_size_t) * 4, hipMemcpyDeviceToHost))
    
    ! Print CSR matrix
    write(*,*) 'CSR'
    do i = 1, m
        start_idx = hcsr_row_ptr(i) + 1
        end_idx = hcsr_row_ptr(i + 1)
        
        temp = 0.0
        do j = start_idx, end_idx
            temp(hcsr_col_ind(j) + 1) = hcsr_val(j)
        end do
        
        do j = 1, n
            write(*,fmt='(F8.1)', advance='no') temp(j)
        end do
        write(*,*)
    end do
    write(*,*)
    
    ! Clean up
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr))
    
    call HIP_CHECK(hipFree(dbsr_row_ptr))
    call HIP_CHECK(hipFree(dbsr_col_ind))
    call HIP_CHECK(hipFree(dbsr_val))
    call HIP_CHECK(hipFree(dcsr_row_ptr))
    call HIP_CHECK(hipFree(dcsr_col_ind))
    call HIP_CHECK(hipFree(dcsr_val))
    
    deallocate(hcsr_row_ptr)
    deallocate(hcsr_col_ind)
    deallocate(hcsr_val)
    deallocate(temp)

end program example_fortran_bsr2csr
! [doc example end]
