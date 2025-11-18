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
program example_fortran_csr2coo
    use iso_c_binding
    use rocsparse

    implicit none

    interface
        function hipMalloc(ptr, size) bind(c, name = 'hipMalloc')
            use iso_c_binding
            integer :: hipMalloc; type(c_ptr) :: ptr; integer(c_size_t), value :: size
        end function
        function hipFree(ptr) bind(c, name = 'hipFree')
            use iso_c_binding
            integer :: hipFree; type(c_ptr), value :: ptr
        end function
        function hipMemcpy(dst, src, size, kind) bind(c, name = 'hipMemcpy')
            use iso_c_binding
            integer :: hipMemcpy; type(c_ptr), value :: dst; type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size; integer(c_int), value :: kind
        end function
    end interface

    integer, parameter :: hipMemcpyHostToDevice = 1, hipMemcpyDeviceToDevice = 2
    
    type(c_ptr) :: handle
    integer :: version
    integer(c_int) :: m, n, nnz
    integer, dimension(4), target :: h_csr_row_ptr
    integer, dimension(8), target :: h_csr_col_ind
    real(c_float), dimension(8), target :: h_csr_val
    type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_coo_row_ind, d_coo_col_ind, d_coo_val

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))

    ! Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)

    m = 3; n = 5; nnz = 8
    h_csr_row_ptr = (/0, 3, 5, 8/)
    h_csr_col_ind = (/0, 1, 3, 1, 2, 0, 3, 4/)
    h_csr_val = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)

    call HIP_CHECK(hipMalloc(d_csr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), int(m + 1, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 4, hipMemcpyHostToDevice))

    call HIP_CHECK(hipMalloc(d_coo_row_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_coo_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_coo_val, int(nnz, c_size_t) * 4))

    call ROCSPARSE_CHECK(rocsparse_csr2coo(handle, d_csr_row_ptr, nnz, m, d_coo_row_ind, rocsparse_index_base_zero))
    call HIP_CHECK(hipMemcpy(d_coo_col_ind, d_csr_col_ind, int(nnz, c_size_t) * 4, hipMemcpyDeviceToDevice))
    call HIP_CHECK(hipMemcpy(d_coo_val, d_csr_val, int(nnz, c_size_t) * 4, hipMemcpyDeviceToDevice))

    call HIP_CHECK(hipFree(d_csr_row_ptr)); call HIP_CHECK(hipFree(d_csr_col_ind)); call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_coo_row_ind)); call HIP_CHECK(hipFree(d_coo_col_ind)); call HIP_CHECK(hipFree(d_coo_val))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2coo
! [doc example end]
