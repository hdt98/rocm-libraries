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
program example_fortran_nnz_compress
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
    real(c_float) :: tol
    
    ! Host arrays
    integer, dimension(4), target :: hcsr_row_ptr_A
    real(c_float), dimension(8), target :: hcsr_val_A
    integer, dimension(3), target :: hnnz_per_row
    
    ! Device pointers
    type(c_ptr) :: dcsr_row_ptr_A, dcsr_val_A, dnnz_per_row
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr_A

    ! Initialize dimensions and tolerance
    m = 3
    n = 5
    nnz_A = 8
    tol = 4.2
    
    ! Initialize host data
    hcsr_row_ptr_A = (/0, 3, 5, 8/)
    hcsr_val_A = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_A))

    ! Allocate device memory
    call HIP_CHECK(hipMalloc(dcsr_row_ptr_A, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val_A, int(nnz_A, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dcsr_row_ptr_A, c_loc(hcsr_row_ptr_A), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_val_A, c_loc(hcsr_val_A), int(nnz_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Allocate memory for the nnz_per_row array
    call HIP_CHECK(hipMalloc(dnnz_per_row, int(m, c_size_t) * 4))

    ! Call snnz_compress() which fills in nnz_per_row array
    call ROCSPARSE_CHECK(rocsparse_snnz_compress(handle, m, descr_A, dcsr_val_A, &
                                                  dcsr_row_ptr_A, dnnz_per_row, c_loc(nnz_C), tol))

    ! Copy result back to host
    call HIP_CHECK(hipMemcpy(c_loc(hnnz_per_row), dnnz_per_row, int(m, c_size_t) * 4, &
                             hipMemcpyDeviceToHost))

    ! Clean up
    call HIP_CHECK(hipFree(dcsr_row_ptr_A))
    call HIP_CHECK(hipFree(dcsr_val_A))
    call HIP_CHECK(hipFree(dnnz_per_row))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_nnz_compress
! [doc example end]
