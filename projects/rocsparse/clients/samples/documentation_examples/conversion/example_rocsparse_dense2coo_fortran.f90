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
program example_fortran_dense2coo
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
    integer, parameter :: hipMemcpyHostToDevice = 1
    
    type(c_ptr) :: handle
    integer :: version
    integer(c_int) :: m, n
    integer(c_int), target :: nnz_A
    real(c_float), dimension(9), target :: hdense
    type(c_ptr) :: ddense, dnnz_per_row, dcoo_row_ind, dcoo_col_ind, dcoo_val, descr

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))

    ! Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)



    m = 3; n = 3
    hdense = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0/)

    call HIP_CHECK(hipMalloc(ddense, int(m * n, c_size_t) * 4))
    call HIP_CHECK(hipMemcpy(ddense, c_loc(hdense), int(m * n, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMalloc(dnnz_per_row, int(m, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_snnz(handle, rocsparse_direction_row, m, n, descr, ddense, m, dnnz_per_row, c_loc(nnz_A)))

    call HIP_CHECK(hipMalloc(dcoo_row_ind, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcoo_col_ind, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcoo_val, int(nnz_A, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_sdense2coo(handle, m, n, descr, ddense, m, dnnz_per_row, &
                                               dcoo_val, dcoo_row_ind, dcoo_col_ind))

    call HIP_CHECK(hipFree(dcoo_row_ind)); call HIP_CHECK(hipFree(dcoo_col_ind))
    call HIP_CHECK(hipFree(dcoo_val)); call HIP_CHECK(hipFree(dnnz_per_row))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_dense2coo
! [doc example end]
