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
program example_fortran_cscsort
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
    integer(c_int) :: m, n, nnz
    
    ! Host arrays
    integer, dimension(4), target :: h_csc_col_ptr
    integer, dimension(9), target :: h_csc_row_ind
    real(c_float), dimension(9), target :: h_csc_val
    
    ! Device pointers
    type(c_ptr) :: d_csc_col_ptr, d_csc_row_ind, d_csc_val
    type(c_ptr) :: d_perm, d_csc_val_sorted, d_temp_buffer
    
    ! rocSPARSE handle
    type(c_ptr) :: handle
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m = 3
    n = 3
    nnz = 9
    
    ! Initialize host data
    h_csc_col_ptr = (/0, 3, 6, 9/)
    h_csc_row_ind = (/2, 0, 1, 0, 1, 2, 0, 2, 1/)
    h_csc_val = (/7.0, 1.0, 4.0, 2.0, 5.0, 8.0, 3.0, 9.0, 6.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Allocate device memory
    call HIP_CHECK(hipMalloc(d_csc_col_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csc_row_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csc_val, int(nnz, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(d_csc_col_ptr, c_loc(h_csc_col_ptr), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csc_row_ind, c_loc(h_csc_row_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csc_val, c_loc(h_csc_val), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create permutation vector perm as the identity map
    call HIP_CHECK(hipMalloc(d_perm, int(nnz, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, d_perm))

    ! Allocate temporary buffer
    call ROCSPARSE_CHECK(rocsparse_cscsort_buffer_size(handle, m, n, nnz, d_csc_col_ptr, &
                                                        d_csc_row_ind, c_loc(buffer_size)))
    call HIP_CHECK(hipMalloc(d_temp_buffer, buffer_size))

    ! Sort the CSC matrix
    call ROCSPARSE_CHECK(rocsparse_cscsort(handle, m, n, nnz, d_csc_col_ptr, &
                                           d_csc_row_ind, d_perm, d_temp_buffer))

    ! Gather sorted csc_val array
    call HIP_CHECK(hipMalloc(d_csc_val_sorted, int(nnz, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_sgthr(handle, nnz, d_csc_val, d_csc_val_sorted, d_perm, &
                                         rocsparse_index_base_zero))

    ! Copy sorted result back to host and print
    block
        integer, allocatable, target :: h_csc_col_ptr(:)
        integer :: i

        allocate(h_csc_col_ptr(n + 1))
        call HIP_CHECK(hipMemcpy(c_loc(h_csc_col_ptr), d_csc_col_ptr, int(n + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,*) 'Sorted CSC matrix:'
        write(*,fmt='(A)',advance='no') 'col_ptr: '
        do i = 1, n + 1
            write(*,fmt='(I0,A)',advance='no') h_csc_col_ptr(i), ' '
        end do
        write(*,*)

        deallocate(h_csc_col_ptr)
    end block

    ! Clean up device memory
    call HIP_CHECK(hipFree(d_temp_buffer))
    call HIP_CHECK(hipFree(d_perm))
    call HIP_CHECK(hipFree(d_csc_val))
    call HIP_CHECK(hipFree(d_csc_val_sorted))
    call HIP_CHECK(hipFree(d_csc_col_ptr))
    call HIP_CHECK(hipFree(d_csc_row_ind))

    ! Destroy rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_cscsort
! [doc example end]
