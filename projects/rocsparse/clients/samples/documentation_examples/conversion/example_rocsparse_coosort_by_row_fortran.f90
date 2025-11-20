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
program example_fortran_coosort_by_row
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
    integer, dimension(9), target :: hcoo_row_ind, hcoo_col_ind
    real(c_float), dimension(9), target :: hcoo_val
    
    ! Device pointers
    type(c_ptr) :: dcoo_row_ind, dcoo_col_ind, dcoo_val
    type(c_ptr) :: perm, dcoo_val_sorted, temp_buffer
    
    ! rocSPARSE handle
    type(c_ptr) :: handle
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m = 3
    n = 3
    nnz = 9
    
    ! Initialize host data (column-major ordering by column)
    hcoo_row_ind = (/0, 1, 2, 0, 1, 2, 0, 1, 2/)
    hcoo_col_ind = (/0, 0, 0, 1, 1, 1, 2, 2, 2/)
    hcoo_val = (/1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0/)

    ! Allocate device memory
    call HIP_CHECK(hipMalloc(dcoo_row_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcoo_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcoo_val, int(nnz, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dcoo_row_ind, c_loc(hcoo_row_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcoo_col_ind, c_loc(hcoo_col_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcoo_val, c_loc(hcoo_val), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create permutation vector perm as the identity map
    call HIP_CHECK(hipMalloc(perm, int(nnz, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, perm))

    ! Allocate temporary buffer
    call ROCSPARSE_CHECK(rocsparse_coosort_buffer_size(handle, m, n, nnz, dcoo_row_ind, &
                                                        dcoo_col_ind, c_loc(buffer_size)))
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Sort the COO matrix by row
    call ROCSPARSE_CHECK(rocsparse_coosort_by_row(handle, m, n, nnz, dcoo_row_ind, &
                                                   dcoo_col_ind, perm, temp_buffer))

    ! Gather sorted coo_val array
    call HIP_CHECK(hipMalloc(dcoo_val_sorted, int(nnz, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_sgthr(handle, nnz, dcoo_val, dcoo_val_sorted, perm, &
                                         rocsparse_index_base_zero))

    ! Copy sorted result back to host and print
    block
        integer, allocatable, target :: hcoo_row_ind_sorted(:), hcoo_col_ind_sorted(:)
        real(c_float), allocatable, target :: hcoo_val_sorted_host(:)
        integer :: i

        allocate(hcoo_row_ind_sorted(nnz))
        allocate(hcoo_col_ind_sorted(nnz))
        allocate(hcoo_val_sorted_host(nnz))

        call HIP_CHECK(hipMemcpy(c_loc(hcoo_row_ind_sorted), dcoo_row_ind, int(nnz, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(hcoo_col_ind_sorted), dcoo_col_ind, int(nnz, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(hcoo_val_sorted_host), dcoo_val_sorted, int(nnz, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,*) 'Sorted COO matrix (by row):'
        do i = 1, nnz
            write(*,fmt='(A,I0,A,I0,A,F0.6)') '(', hcoo_row_ind_sorted(i), ', ', &
                hcoo_col_ind_sorted(i), '): ', hcoo_val_sorted_host(i)
        end do

        deallocate(hcoo_row_ind_sorted)
        deallocate(hcoo_col_ind_sorted)
        deallocate(hcoo_val_sorted_host)
    end block

    ! Clear rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

    ! Clean up device memory
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(perm))
    call HIP_CHECK(hipFree(dcoo_row_ind))
    call HIP_CHECK(hipFree(dcoo_col_ind))
    call HIP_CHECK(hipFree(dcoo_val))
    call HIP_CHECK(hipFree(dcoo_val_sorted))

end program example_fortran_coosort_by_row
! [doc example end]
