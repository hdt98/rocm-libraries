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
program example_fortran_prune_csr2csr
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
    integer(c_int) :: m, n, nnz_A
    integer(c_int), target :: nnz_C
    real(c_float), target :: threshold
    
    ! Host arrays
    integer, dimension(5), target :: hcsr_row_ptr_A
    integer, dimension(11), target :: hcsr_col_ind_A
    real(c_float), dimension(11), target :: hcsr_val_A
    
    ! Device pointers
    type(c_ptr) :: dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A
    type(c_ptr) :: dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr_A, descr_C, info
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m = 4
    n = 4
    nnz_A = 11
    threshold = 5.0
    
    ! Initialize host data
    hcsr_row_ptr_A = (/0, 2, 4, 7, 11/)
    hcsr_col_ind_A = (/0, 1, 0, 3, 0, 1, 3, 0, 1, 2, 3/)
    hcsr_val_A = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 4.0, 2.0, 5.0/)

    ! Allocate device memory for input CSR matrix
    call HIP_CHECK(hipMalloc(dcsr_row_ptr_A, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_col_ind_A, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val_A, int(nnz_A, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dcsr_row_ptr_A, c_loc(hcsr_row_ptr_A), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_col_ind_A, c_loc(hcsr_col_ind_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_val_A, c_loc(hcsr_val_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_C))
    call ROCSPARSE_CHECK(rocsparse_create_mat_info(info))

    ! Allocate device memory for output CSR row pointer
    call HIP_CHECK(hipMalloc(dcsr_row_ptr_C, int(m + 1, c_size_t) * 4))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_sprune_csr2csr_buffer_size(handle, m, n, nnz_A, descr_A, &
                                                               dcsr_val_A, dcsr_row_ptr_A, &
                                                               dcsr_col_ind_A, c_loc(threshold), &
                                                               descr_C, c_null_ptr, dcsr_row_ptr_C, &
                                                               c_null_ptr, c_loc(buffer_size)))

    ! Allocate temporary buffer
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Compute nnz_C
    call ROCSPARSE_CHECK(rocsparse_sprune_csr2csr_nnz(handle, m, n, nnz_A, descr_A, dcsr_val_A, &
                                                       dcsr_row_ptr_A, dcsr_col_ind_A, &
                                                       c_loc(threshold), descr_C, dcsr_row_ptr_C, &
                                                       c_loc(nnz_C), temp_buffer))

    ! Allocate device memory for output CSR matrix
    call HIP_CHECK(hipMalloc(dcsr_col_ind_C, int(nnz_C, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val_C, int(nnz_C, c_size_t) * 4))

    ! Perform pruning
    call ROCSPARSE_CHECK(rocsparse_sprune_csr2csr(handle, m, n, nnz_A, descr_A, dcsr_val_A, &
                                                   dcsr_row_ptr_A, dcsr_col_ind_A, &
                                                   c_loc(threshold), descr_C, dcsr_val_C, &
                                                   dcsr_row_ptr_C, dcsr_col_ind_C, temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: hcsr_row_ptr_C(:)
        integer :: i

        allocate(hcsr_row_ptr_C(m + 1))
        call HIP_CHECK(hipMemcpy(c_loc(hcsr_row_ptr_C), dcsr_row_ptr_C, int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'nnz_C after pruning: ', nnz_C
        write(*,fmt='(A)',advance='no') 'CSR row_ptr: '
        do i = 1, m + 1
            write(*,fmt='(I0,A)',advance='no') hcsr_row_ptr_C(i), ' '
        end do
        write(*,*)

        deallocate(hcsr_row_ptr_C)
    end block

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C))

    ! Clean up device memory
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(dcsr_row_ptr_A))
    call HIP_CHECK(hipFree(dcsr_col_ind_A))
    call HIP_CHECK(hipFree(dcsr_val_A))
    call HIP_CHECK(hipFree(dcsr_row_ptr_C))
    call HIP_CHECK(hipFree(dcsr_col_ind_C))
    call HIP_CHECK(hipFree(dcsr_val_C))

end program example_fortran_prune_csr2csr
! [doc example end]
