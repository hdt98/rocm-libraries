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
program example_fortran_gebsr2gebsr
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
    integer(c_int) :: mb_A, nb_A, nnzb_A, row_block_dim_A, col_block_dim_A
    integer(c_int) :: mb_C, nb_C, row_block_dim_C, col_block_dim_C
    integer(c_int) :: m, n, dir
    integer(c_int), target :: nnzb_C
    
    ! Host arrays
    integer, dimension(3), target :: hbsr_row_ptr_A
    integer, dimension(4), target :: hbsr_col_ind_A
    real(c_float), dimension(16), target :: hbsr_val_A
    
    ! Device pointers
    type(c_ptr) :: dbsr_row_ptr_A, dbsr_col_ind_A, dbsr_val_A
    type(c_ptr) :: dbsr_row_ptr_C, dbsr_col_ind_C, dbsr_val_C
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr_A, descr_C
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size

    ! Initialize dimensions
    mb_A = 2
    nb_A = 2
    nnzb_A = 4
    row_block_dim_A = 2
    col_block_dim_A = 2
    
    m = mb_A * row_block_dim_A
    n = nb_A * col_block_dim_A
    
    dir = rocsparse_direction_row
    
    ! Initialize host data
    hbsr_row_ptr_A = (/0, 2, 4/)
    hbsr_col_ind_A = (/0, 2, 0, 1/)
    hbsr_val_A = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, &
                   6.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0/)

    ! Allocate device memory for input BSR matrix
    call HIP_CHECK(hipMalloc(dbsr_row_ptr_A, int(mb_A + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_col_ind_A, int(nnzb_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val_A, &
                             int(nnzb_A * row_block_dim_A * col_block_dim_A, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dbsr_row_ptr_A, c_loc(hbsr_row_ptr_A), int(mb_A + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_col_ind_A, c_loc(hbsr_col_ind_A), int(nnzb_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dbsr_val_A, c_loc(hbsr_val_A), &
                             int(nnzb_A * row_block_dim_A * col_block_dim_A, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Define output block dimensions
    row_block_dim_C = 2
    col_block_dim_C = 3
    mb_C = (m + row_block_dim_C - 1) / row_block_dim_C
    nb_C = (m + row_block_dim_C - 1) / row_block_dim_C

    ! Allocate device memory for output BSR row pointer
    call HIP_CHECK(hipMalloc(dbsr_row_ptr_C, int(mb_C + 1, c_size_t) * 4))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_C))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_sgebsr2gebsr_buffer_size(handle, dir, mb_A, nb_A, nnzb_A, &
                                                             descr_A, dbsr_val_A, dbsr_row_ptr_A, &
                                                             dbsr_col_ind_A, row_block_dim_A, &
                                                             col_block_dim_A, row_block_dim_C, &
                                                             col_block_dim_C, c_loc(buffer_size)))

    ! Allocate temporary buffer
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Compute nnzb_C
    call ROCSPARSE_CHECK(rocsparse_gebsr2gebsr_nnz(handle, dir, mb_A, nb_A, nnzb_A, descr_A, &
                                                    dbsr_row_ptr_A, dbsr_col_ind_A, &
                                                    row_block_dim_A, col_block_dim_A, descr_C, &
                                                    dbsr_row_ptr_C, row_block_dim_C, &
                                                    col_block_dim_C, c_loc(nnzb_C), temp_buffer))

    ! Allocate device memory for output BSR matrix
    call HIP_CHECK(hipMalloc(dbsr_col_ind_C, int(nnzb_C, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dbsr_val_C, &
                             int(nnzb_C * row_block_dim_C * col_block_dim_C, c_size_t) * 4))

    ! Perform conversion
    call ROCSPARSE_CHECK(rocsparse_sgebsr2gebsr(handle, dir, mb_A, nb_A, nnzb_A, descr_A, &
                                                 dbsr_val_A, dbsr_row_ptr_A, dbsr_col_ind_A, &
                                                 row_block_dim_A, col_block_dim_A, descr_C, &
                                                 dbsr_val_C, dbsr_row_ptr_C, dbsr_col_ind_C, &
                                                 row_block_dim_C, col_block_dim_C, temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: h_bsr_row_ptr_C(:)
        integer :: i

        allocate(h_bsr_row_ptr_C(mb_C + 1))
        call HIP_CHECK(hipMemcpy(c_loc(h_bsr_row_ptr_C), dbsr_row_ptr_C, &
            int(mb_C + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A,I0)') 'nnzb_C: ', nnzb_C
        write(*,fmt='(A)',advance='no') 'GEBSR_C row_ptr: '
        do i = 1, mb_C + 1
            write(*,fmt='(I0,A)',advance='no') h_bsr_row_ptr_C(i), ' '
        end do
        write(*,*)

        deallocate(h_bsr_row_ptr_C)
    end block

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C))

    ! Clean up device memory
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(dbsr_row_ptr_A))
    call HIP_CHECK(hipFree(dbsr_col_ind_A))
    call HIP_CHECK(hipFree(dbsr_val_A))
    call HIP_CHECK(hipFree(dbsr_row_ptr_C))
    call HIP_CHECK(hipFree(dbsr_col_ind_C))
    call HIP_CHECK(hipFree(dbsr_val_C))

end program example_fortran_gebsr2gebsr
! [doc example end]
