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
program example_fortran_csr2hyb
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
    integer(c_int) :: m, n, nnz, user_ell_width
    
    ! Host arrays
    integer, dimension(5), target :: hcsr_row_ptr
    integer, dimension(12), target :: hcsr_col_ind
    real(c_float), dimension(12), target :: hcsr_val
    
    ! Device pointers
    type(c_ptr) :: dcsr_row_ptr, dcsr_col_ind, dcsr_val
    type(c_ptr) :: dcsr_row_ptr2, dcsr_col_ind2, dcsr_val2
    type(c_ptr) :: temp_buffer
    
    ! rocSPARSE handles
    type(c_ptr) :: handle, descr, hyb
    
    ! Buffer size
    integer(c_size_t), target :: buffer_size
    
    ! Partition type
    integer(c_int) :: partition_type

    ! Initialize dimensions
    m = 4
    n = 6
    nnz = 12
    user_ell_width = 3
    partition_type = rocsparse_hyb_partition_user
    
    ! Initialize host data
    hcsr_row_ptr = (/0, 4, 6, 10, 12/)
    hcsr_col_ind = (/0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1/)
    hcsr_val = (/1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 6.0, 5.0, 3.0, 4.0, 1.0, 2.0/)

    ! Allocate device memory for CSR format
    call HIP_CHECK(hipMalloc(dcsr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val, int(nnz, c_size_t) * 4))

    ! Copy data to device
    call HIP_CHECK(hipMemcpy(dcsr_row_ptr, c_loc(hcsr_row_ptr), int(m + 1, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_col_ind, c_loc(hcsr_col_ind), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))
    call HIP_CHECK(hipMemcpy(dcsr_val, c_loc(hcsr_val), int(nnz, c_size_t) * 4, &
                             hipMemcpyHostToDevice))

    ! Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

    ! Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr))

    ! Create HYB matrix
    call ROCSPARSE_CHECK(rocsparse_create_hyb_mat(hyb))

    ! Convert CSR to HYB
    call ROCSPARSE_CHECK(rocsparse_scsr2hyb(handle, m, n, descr, dcsr_val, dcsr_row_ptr, &
                                            dcsr_col_ind, hyb, user_ell_width, partition_type))

    ! Allocate device memory for output CSR format
    call HIP_CHECK(hipMalloc(dcsr_row_ptr2, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_col_ind2, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dcsr_val2, int(nnz, c_size_t) * 4))

    ! Obtain the temporary buffer size
    call ROCSPARSE_CHECK(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr2, &
                                                        c_loc(buffer_size)))

    ! Allocate temporary buffer
    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

    ! Convert HYB to CSR
    call ROCSPARSE_CHECK(rocsparse_shyb2csr(handle, descr, hyb, dcsr_val2, dcsr_row_ptr2, &
                                            dcsr_col_ind2, temp_buffer))

    ! Copy result back to host and print
    block
        integer, allocatable, target :: hcsr_row_ptr2(:)
        integer :: i

        allocate(hcsr_row_ptr2(m + 1))
        call HIP_CHECK(hipMemcpy(c_loc(hcsr_row_ptr2), dcsr_row_ptr2, &
            int(m + 1, c_size_t) * 4, hipMemcpyDeviceToHost))

        write(*,fmt='(A)',advance='no') 'Converted CSR row_ptr (HYB->CSR): '
        do i = 1, m + 1
            write(*,fmt='(I0,A)',advance='no') hcsr_row_ptr2(i), ' '
        end do
        write(*,*)

        deallocate(hcsr_row_ptr2)
    end block

    ! Clean up
    call HIP_CHECK(hipFree(temp_buffer))
    call HIP_CHECK(hipFree(dcsr_row_ptr))
    call HIP_CHECK(hipFree(dcsr_col_ind))
    call HIP_CHECK(hipFree(dcsr_val))
    call HIP_CHECK(hipFree(dcsr_row_ptr2))
    call HIP_CHECK(hipFree(dcsr_col_ind2))
    call HIP_CHECK(hipFree(dcsr_val2))

    ! Destroy rocSPARSE handles
    call ROCSPARSE_CHECK(rocsparse_destroy_hyb_mat(hyb))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

end program example_fortran_csr2hyb
! [doc example end]
