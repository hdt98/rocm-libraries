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
program example_fortran_csr2csr_compress
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

    ! Matrix dimensions
    integer(c_int) :: m, n, nnz_A
    integer(c_int), target :: nnz_C
    real(c_float) :: tol
    
    ! Host arrays
    integer, dimension(4), target :: h_csr_row_ptr_A
    integer, dimension(8), target :: h_csr_col_ind_A
    real(c_float), dimension(8), target :: h_csr_val_A
    
    ! Device pointers
    type(c_ptr) :: d_csr_row_ptr_A, d_csr_col_ind_A, d_csr_val_A
    type(c_ptr) :: d_csr_row_ptr_C, d_csr_col_ind_C, d_csr_val_C
    type(c_ptr) :: d_nnz_per_row
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr

    ! Initialize dimensions and tolerance
    m = 3
    n = 5
    nnz_A = 8
    tol = 0.0
    
    ! Initialize host data
    h_csr_row_ptr_A = (/0, 3, 5, 8/)
    h_csr_col_ind_A = (/0, 1, 3, 1, 2, 0, 3, 4/)
    h_csr_val_A = (/1.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 0.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Allocate device memory for matrix A
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_A, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_A, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_A, int(nnz_A, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_A, c_loc(h_csr_row_ptr_A), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_A, c_loc(h_csr_col_ind_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val_A, c_loc(h_csr_val_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Allocate memory for compressed CSR matrix
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_C, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_nnz_per_row, int(m, c_size_t) * 4))

    ! Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr))

    ! Call nnz_compress() which fills in nnz_per_row array
    call ROCSPARSE_CHECK(rocsparse_snnz_compress(handle, m, descr, d_csr_val_A, &
                                                  d_csr_row_ptr_A, d_nnz_per_row, c_loc(nnz_C), tol))

    ! Allocate column indices and values array for compressed CSR matrix
    call HIP_CHECK(hipMalloc(d_csr_col_ind_C, int(nnz_C, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_C, int(nnz_C, c_size_t) * 4))

    ! Finish compression by calling csr2csr_compress()
    call ROCSPARSE_CHECK(rocsparse_scsr2csr_compress(handle, m, n, descr, d_csr_val_A, &
                                                      d_csr_row_ptr_A, d_csr_col_ind_A, nnz_A, &
                                                      d_nnz_per_row, d_csr_val_C, d_csr_row_ptr_C, &
                                                      d_csr_col_ind_C, tol))

    ! Clean up
    call HIP_CHECK(hipFree(d_csr_row_ptr_A))
    call HIP_CHECK(hipFree(d_csr_col_ind_A))
    call HIP_CHECK(hipFree(d_csr_val_A))
    call HIP_CHECK(hipFree(d_csr_row_ptr_C))
    call HIP_CHECK(hipFree(d_csr_col_ind_C))
    call HIP_CHECK(hipFree(d_csr_val_C))
    call HIP_CHECK(hipFree(d_nnz_per_row))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2csr_compress
! [doc example end]
