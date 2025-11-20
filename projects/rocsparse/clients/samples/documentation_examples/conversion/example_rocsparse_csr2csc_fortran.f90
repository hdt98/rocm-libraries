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
program example_fortran_csr2csc
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

    integer, parameter :: hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2

    ! Matrix dimensions
    integer(c_int) :: m_A, n_A, nnz_A, m_T, n_T, nnz_T
    integer :: i
    
    ! Host arrays
    integer, dimension(4), target :: h_csr_row_ptr_A
    integer, dimension(8), target :: h_csr_col_ind_A
    real(c_float), dimension(8), target :: h_csr_val_A
    
    ! Device pointers
    type(c_ptr) :: d_csr_row_ptr_A, d_csr_col_ind_A, d_csr_val_A
    type(c_ptr) :: d_csr_row_ptr_T, d_csr_col_ind_T, d_csr_val_T
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handle
    type(c_ptr) :: handle
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    m_A = 3
    n_A = 5
    nnz_A = 8
    
    ! Initialize host data
    h_csr_row_ptr_A = (/0, 3, 5, 8/)
    h_csr_col_ind_A = (/0, 1, 3, 1, 2, 0, 3, 4/)
    h_csr_val_A = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)

    ! Allocate device memory for matrix A
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_A, int(m_A + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_A, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_A, int(nnz_A, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_A, c_loc(h_csr_row_ptr_A), int(m_A + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_A, c_loc(h_csr_col_ind_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val_A, c_loc(h_csr_val_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Allocate memory for transposed CSR matrix (CSC format)
    m_T = n_A
    n_T = m_A
    nnz_T = nnz_A
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_T, int(m_T + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_T, int(nnz_T, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_T, int(nnz_T, c_size_t) * 4))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle, m_A, n_A, nnz_A, &
                                                        d_csr_row_ptr_A, d_csr_col_ind_A, &
                                                        rocsparse_action_numeric, &
                                                        c_loc(buffer_size)))
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Perform the CSR to CSC conversion
    call ROCSPARSE_CHECK(rocsparse_scsr2csc(handle, m_A, n_A, nnz_A, d_csr_val_A, &
                                            d_csr_row_ptr_A, d_csr_col_ind_A, d_csr_val_T, &
                                            d_csr_col_ind_T, d_csr_row_ptr_T, &
                                            rocsparse_action_numeric, rocsparse_index_base_zero, &
                                            temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: h_csc_col_ptr(:), h_csc_row_ind(:)
        real(c_float), allocatable, target :: h_csc_val(:)

        allocate(h_csc_col_ptr(m_T + 1))
        allocate(h_csc_row_ind(nnz_T))
        allocate(h_csc_val(nnz_T))

        call HIP_CHECK(hipMemcpy(c_loc(h_csc_col_ptr), d_csr_row_ptr_T, int(m_T + 1, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(h_csc_row_ind), d_csr_col_ind_T, int(nnz_T, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(h_csc_val), d_csr_val_T, int(nnz_T, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A)',advance='no') 'CSC col_ptr: '
        do i = 1, m_T + 1
            write(*,fmt='(I0,A)',advance='no') h_csc_col_ptr(i), ' '
        end do
        write(*,*)

        write(*,fmt='(A)',advance='no') 'CSC row_ind: '
        do i = 1, nnz_T
            write(*,fmt='(I0,A)',advance='no') h_csc_row_ind(i), ' '
        end do
        write(*,*)

        write(*,fmt='(A)',advance='no') 'CSC val: '
        do i = 1, nnz_T
            write(*,fmt='(F0.1,A)',advance='no') h_csc_val(i), ' '
        end do
        write(*,*)

        deallocate(h_csc_col_ptr)
        deallocate(h_csc_row_ind)
        deallocate(h_csc_val)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(d_csr_row_ptr_A))
    call HIP_CHECK(hipFree(d_csr_col_ind_A))
    call HIP_CHECK(hipFree(d_csr_val_A))
    call HIP_CHECK(hipFree(d_csr_row_ptr_T))
    call HIP_CHECK(hipFree(d_csr_col_ind_T))
    call HIP_CHECK(hipFree(d_csr_val_T))
    call HIP_CHECK(hipFree(temp_buffer))

    ! Destroy rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2csc
! [doc example end]
