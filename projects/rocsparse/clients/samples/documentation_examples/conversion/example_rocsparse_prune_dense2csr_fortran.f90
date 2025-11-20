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
program example_fortran_prune_dense2csr
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
    integer(c_int) :: m, n, lda
    integer(c_int), target :: nnz
    real(c_float), target :: threshold
    
    ! Host array
    real(c_float), dimension(16), target :: hdense
    
    ! Device pointers
    type(c_ptr) :: ddense
    type(c_ptr) :: dcsr_row_ptr, dcsr_col_ind, dcsr_val
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr, info
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m = 4
    n = 4
    lda = m
    threshold = 3.0
    
    ! Initialize host data (column-major order)
    hdense = (/1.0, 3.0, 5.0, 0.0, 2.0, 0.0, 6.0, 4.0, &
               0.0, 0.0, 0.0, 2.0, 7.0, 4.0, 4.0, 5.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_create_mat_info(info))

    ! Allocate device memory for dense matrix
    call HIP_CHECK(hipMalloc(ddense, int(lda * n, c_size_t) * 4))
    call HIP_CHECK(hipMemcpy(ddense, c_loc(hdense), int(lda * n, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Allocate device memory for CSR row pointer
    call HIP_CHECK(hipMalloc(dcsr_row_ptr, int(m + 1, c_size_t) * 4))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_sprune_dense2csr_buffer_size(handle, m, n, ddense, lda, &
                                                                 c_loc(threshold), descr, &
                                                                 c_null_ptr, dcsr_row_ptr, &
                                                                 c_null_ptr, c_loc(buffer_size)))

    ! Allocate temporary buffer
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Compute nnz
    call ROCSPARSE_CHECK(rocsparse_sprune_dense2csr_nnz(handle, m, n, ddense, lda, &
                                                         c_loc(threshold), descr, dcsr_row_ptr, &
                                                         c_loc(nnz), temp_buffer))

    ! Allocate device memory for CSR matrix
    call HIP_CHECK(hipMalloc(dcsr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val, int(nnz, c_size_t) * 4))

    ! Perform pruning
    call ROCSPARSE_CHECK(rocsparse_sprune_dense2csr(handle, m, n, ddense, lda, c_loc(threshold), &
                                                     descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, &
                                                     temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: hcsr_row_ptr(:)
        integer :: i

        allocate(hcsr_row_ptr(m + 1))
        call HIP_CHECK(hipMemcpy(c_loc(hcsr_row_ptr), dcsr_row_ptr, &
            int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'nnz after pruning: ', nnz
        write(*,fmt='(A)',advance='no') 'CSR row_ptr: '
        do i = 1, m + 1
            write(*,fmt='(I0,A)',advance='no') hcsr_row_ptr(i), ' '
        end do
        write(*,*)

        deallocate(hcsr_row_ptr)
    end block

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info))

    ! Clean up device memory
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(ddense))
    call HIP_CHECK(hipFree(dcsr_row_ptr))
    call HIP_CHECK(hipFree(dcsr_col_ind))
    call HIP_CHECK(hipFree(dcsr_val))

end program example_fortran_prune_dense2csr
! [doc example end]
