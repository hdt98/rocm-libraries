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
program example_fortran_csrgeam
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
    integer(c_int) :: m, n, nnz_A, nnz_B
    integer(c_int), target :: nnz_C
    
    ! Host arrays
    integer, dimension(4), target :: h_csr_row_ptr_A, h_csr_col_ind_A
    integer, dimension(4), target :: h_csr_row_ptr_B, h_csr_col_ind_B
    real(c_float), dimension(4), target :: h_csr_val_A, h_csr_val_B
    real(c_float), target :: alpha, beta
    
    ! Device pointers
    type(c_ptr) :: d_csr_row_ptr_A, d_csr_col_ind_A, d_csr_val_A
    type(c_ptr) :: d_csr_row_ptr_B, d_csr_col_ind_B, d_csr_val_B
    type(c_ptr) :: d_csr_row_ptr_C, d_csr_col_ind_C, d_csr_val_C
    
    ! rocSPARSE handles
    type(c_ptr) :: handle
    type(c_ptr) :: descr_A, descr_B, descr_C
    
    ! Version
    integer :: version

    ! Initialize values
    m = 3
    n = 3
    nnz_A = 4
    nnz_B = 4
    alpha = 1.0
    beta = 1.0
    h_csr_row_ptr_A = (/0, 2, 3, 4/)
    h_csr_col_ind_A = (/0, 1, 2, 2/)
    h_csr_val_A = (/1.0, 2.0, 3.0, 4.0/)
    h_csr_row_ptr_B = (/0, 1, 3, 4/)
    h_csr_col_ind_B = (/0, 1, 2, 2/)
    h_csr_val_B = (/5.0, 6.0, 7.0, 8.0/)

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))

    ! Print version on screen
    ! Commented out to avoid contaminating numerical output
    ! write(*,fmt='(A,I0,A,I0,A,I0)') 'rocSPARSE version: ', version / 100000, '.', &
    !     mod(version / 100, 1000), '.', mod(version, 100)

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_B))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_C))
    call ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host))

    call HIP_CHECK(hipMalloc(d_csr_row_ptr_A, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_A, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_A, int(nnz_A, c_size_t) * 4))
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_A, c_loc(h_csr_row_ptr_A), int(m + 1, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_A, c_loc(h_csr_col_ind_A), int(nnz_A, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val_A, c_loc(h_csr_val_A), int(nnz_A, c_size_t) * 4, hipMemcpyHostToDevice))

    call HIP_CHECK(hipMalloc(d_csr_row_ptr_B, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_B, int(nnz_B, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_B, int(nnz_B, c_size_t) * 4))
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_B, c_loc(h_csr_row_ptr_B), int(m + 1, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_B, c_loc(h_csr_col_ind_B), int(nnz_B, c_size_t) * 4, hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(d_csr_val_B, c_loc(h_csr_val_B), int(nnz_B, c_size_t) * 4, hipMemcpyHostToDevice))

    call HIP_CHECK(hipMalloc(d_csr_row_ptr_C, int(m + 1, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_csrgeam_nnz(handle, m, n, descr_A, nnz_A, d_csr_row_ptr_A, d_csr_col_ind_A, &
                                                 descr_B, nnz_B, d_csr_row_ptr_B, d_csr_col_ind_B, descr_C, &
                                                 d_csr_row_ptr_C, c_loc(nnz_C)))

    call HIP_CHECK(hipMalloc(d_csr_col_ind_C, int(nnz_C, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_C, int(nnz_C, c_size_t) * 4))
    call ROCSPARSE_CHECK(rocsparse_scsrgeam(handle, m, n, c_loc(alpha), descr_A, nnz_A, d_csr_val_A, &
                                             d_csr_row_ptr_A, d_csr_col_ind_A, c_loc(beta), descr_B, nnz_B, &
                                             d_csr_val_B, d_csr_row_ptr_B, d_csr_col_ind_B, descr_C, d_csr_val_C, &
                                             d_csr_row_ptr_C, d_csr_col_ind_C))

    write(*,fmt='(A,I0)') 'nnz_C = ', nnz_C

    ! Copy C matrix result back to host and print
    block
        integer, allocatable, target :: h_csr_row_ptr_C(:), h_csr_col_ind_C(:)
        real(c_float), allocatable, target :: h_csr_val_C(:)
        real(c_float), allocatable :: h_temp(:)
        integer :: i, j, start_idx, end_idx

        allocate(h_csr_row_ptr_C(m + 1))
        allocate(h_csr_col_ind_C(nnz_C))
        allocate(h_csr_val_C(nnz_C))

        call HIP_CHECK(hipMemcpy(c_loc(h_csr_row_ptr_C), d_csr_row_ptr_C, int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(h_csr_col_ind_C), d_csr_col_ind_C, int(nnz_C, c_size_t) * 4, hipMemcpyDeviceToHost))
        call HIP_CHECK(hipMemcpy(c_loc(h_csr_val_C), d_csr_val_C, int(nnz_C, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,*) 'C'
        do i = 1, m
            start_idx = h_csr_row_ptr_C(i) + 1
            end_idx = h_csr_row_ptr_C(i + 1)

            allocate(h_temp(n))
            h_temp = 0.0

            do j = start_idx, end_idx
                h_temp(h_csr_col_ind_C(j) + 1) = h_csr_val_C(j)
            end do

            do j = 1, n
                write(*,fmt='(F0.1,A)',advance='no') h_temp(j), ' '
            end do
            write(*,*)

            deallocate(h_temp)
        end do
        write(*,*)

        deallocate(h_csr_row_ptr_C)
        deallocate(h_csr_col_ind_C)
        deallocate(h_csr_val_C)
    end block

    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
    call HIP_CHECK(hipFree(d_csr_row_ptr_A)); call HIP_CHECK(hipFree(d_csr_col_ind_A)); call HIP_CHECK(hipFree(d_csr_val_A))
    call HIP_CHECK(hipFree(d_csr_row_ptr_B)); call HIP_CHECK(hipFree(d_csr_col_ind_B)); call HIP_CHECK(hipFree(d_csr_val_B))
    call HIP_CHECK(hipFree(d_csr_row_ptr_C)); call HIP_CHECK(hipFree(d_csr_col_ind_C)); call HIP_CHECK(hipFree(d_csr_val_C))

end program example_fortran_csrgeam
! [doc example end]
