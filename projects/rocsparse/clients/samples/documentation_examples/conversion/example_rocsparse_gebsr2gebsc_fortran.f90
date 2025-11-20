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
program example_fortran_gebsr2gebsc
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
    integer(c_int) :: mb_A, nb_A, nnzb_A, row_block_dim, col_block_dim
    integer(c_int) :: mb_T, nb_T, nnzb_T
    
    ! Host arrays
    integer, dimension(3), target :: hbsr_row_ptr_A
    integer, dimension(4), target :: hbsr_col_ind_A
    real(c_float), dimension(16), target :: hbsr_val_A
    
    ! Device pointers
    type(c_ptr) :: dbsr_row_ptr_A, dbsr_col_ind_A, dbsr_val_A
    type(c_ptr) :: dbsr_row_ptr_T, dbsr_col_ind_T, dbsr_val_T
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handle
    type(c_ptr) :: handle
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    mb_A = 2
    nb_A = 2
    nnzb_A = 4
    row_block_dim = 2
    col_block_dim = 2
    
    ! Initialize host data
    hbsr_row_ptr_A = (/0, 2, 4/)
    hbsr_col_ind_A = (/0, 1, 0, 1/)
    hbsr_val_A = (/1.0, 2.0, 0.0, 4.0, 0.0, 3.0, 5.0, 0.0, &
                   6.0, 0.0, 1.0, 2.0, 0.0, 7.0, 3.0, 4.0/)

    ! Allocate device memory for input BSR matrix
    call HIP_CHECK(hipMalloc(dbsr_row_ptr_A, int(mb_A + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_col_ind_A, int(nnzb_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val_A, &
                             int(nnzb_A * row_block_dim * col_block_dim, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dbsr_row_ptr_A, c_loc(hbsr_row_ptr_A), int(mb_A + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_col_ind_A, c_loc(hbsr_col_ind_A), int(nnzb_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_val_A, c_loc(hbsr_val_A), &
                             int(nnzb_A * row_block_dim * col_block_dim, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Allocate memory for transposed BSR matrix
    mb_T = nb_A
    nb_T = mb_A
    nnzb_T = nnzb_A

    call HIP_CHECK(hipMalloc(dbsr_row_ptr_T, int(mb_T + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_col_ind_T, int(nnzb_T, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val_T, &
                             int(nnzb_A * row_block_dim * col_block_dim, c_size_t) * 4))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_sgebsr2gebsc_buffer_size(handle, mb_A, nb_A, nnzb_A, &
                                                             dbsr_val_A, dbsr_row_ptr_A, &
                                                             dbsr_col_ind_A, row_block_dim, &
                                                             col_block_dim, c_loc(buffer_size)))

    ! Allocate temporary buffer
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Perform transpose
    call ROCSPARSE_CHECK(rocsparse_sgebsr2gebsc(handle, mb_A, nb_A, nnzb_A, dbsr_val_A, &
                                                 dbsr_row_ptr_A, dbsr_col_ind_A, row_block_dim, &
                                                 col_block_dim, dbsr_val_T, dbsr_col_ind_T, &
                                                 dbsr_row_ptr_T, rocsparse_action_numeric, &
                                                 rocsparse_index_base_zero, temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: hbsr_row_ptr_T(:)
        integer :: i

        allocate(hbsr_row_ptr_T(mb_T + 1))
        call HIP_CHECK(hipMemcpy(c_loc(hbsr_row_ptr_T), dbsr_row_ptr_T, &
            int(mb_T + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A)',advance='no') 'GEBSC col_ptr: '
        do i = 1, mb_T + 1
            write(*,fmt='(I0,A)',advance='no') hbsr_row_ptr_T(i), ' '
        end do
        write(*,*)

        deallocate(hbsr_row_ptr_T)
    end block

    ! Clear rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

    ! Clean up device memory
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(dbsr_row_ptr_A))
    call HIP_CHECK(hipFree(dbsr_col_ind_A))
    call HIP_CHECK(hipFree(dbsr_val_A))
    call HIP_CHECK(hipFree(dbsr_row_ptr_T))
    call HIP_CHECK(hipFree(dbsr_col_ind_T))
    call HIP_CHECK(hipFree(dbsr_val_T))

end program example_fortran_gebsr2gebsc
! [doc example end]
