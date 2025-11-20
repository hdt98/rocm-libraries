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
program example_fortran_ell2csr
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
    integer(c_int) :: m, n, nnz, ell_width
    integer(c_int), target :: csr_nnz
    
    ! Host arrays
    integer, dimension(9), target :: hell_col_ind
    real(c_float), dimension(9), target :: hell_val
    
    ! Device pointers
    type(c_ptr) :: dell_col_ind, dell_val
    type(c_ptr) :: dcsr_row_ptr, dcsr_col_ind, dcsr_val
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, ell_descr, csr_descr

    ! Initialize dimensions
    m = 3
    n = 5
    nnz = 8
    ell_width = 3
    
    ! Initialize host data
    hell_col_ind = (/0, 1, 0, 1, 2, 3, 3, -1, 4/)
    hell_val = (/1.0, 4.0, 6.0, 2.0, 5.0, 7.0, 3.0, 0.0, 8.0/)

    ! Allocate device memory for ELL format
    call HIP_CHECK(hipMalloc(dell_col_ind, int(m * ell_width, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dell_val, int(m * ell_width, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dell_col_ind, c_loc(hell_col_ind), int(m * ell_width, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dell_val, c_loc(hell_val), int(m * ell_width, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create ELL matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(ell_descr))

    ! Create CSR matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(csr_descr))

    ! Allocate csr_row_ptr array for row offsets
    call HIP_CHECK(hipMalloc(dcsr_row_ptr, int(m + 1, c_size_t) * 4))

    ! Obtain the number of CSR non-zero entries and fill csr_row_ptr array
    call ROCSPARSE_CHECK(rocsparse_ell2csr_nnz(handle, m, n, ell_descr, ell_width, &
                                                dell_col_ind, csr_descr, dcsr_row_ptr, &
                                                c_loc(csr_nnz)))

    ! Allocate CSR column and value arrays
    call HIP_CHECK(hipMalloc(dcsr_col_ind, int(csr_nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val, int(csr_nnz, c_size_t) * 4))

    ! Format conversion
    call ROCSPARSE_CHECK(rocsparse_sell2csr(handle, m, n, ell_descr, ell_width, dell_val, &
                                            dell_col_ind, csr_descr, dcsr_val, dcsr_row_ptr, &
                                            dcsr_col_ind))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: hcsr_row_ptr(:)
        integer :: i

        allocate(hcsr_row_ptr(m + 1))
        call HIP_CHECK(hipMemcpy(c_loc(hcsr_row_ptr), dcsr_row_ptr, &
            int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'csr_nnz: ', csr_nnz
        write(*,fmt='(A)',advance='no') 'CSR row_ptr: '
        do i = 1, m + 1
            write(*,fmt='(I0,A)',advance='no') hcsr_row_ptr(i), ' '
        end do
        write(*,*)

        deallocate(hcsr_row_ptr)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(dell_col_ind))
    call HIP_CHECK(hipFree(dell_val))
    call HIP_CHECK(hipFree(dcsr_row_ptr))
    call HIP_CHECK(hipFree(dcsr_col_ind))
    call HIP_CHECK(hipFree(dcsr_val))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(ell_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_ell2csr
! [doc example end]
