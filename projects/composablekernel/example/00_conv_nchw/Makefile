cflags+=-g -O3 -std=gnu++20 -Wno-unused-value \
	-Wno-undefined-var-template \
	-fno-offload-uniform-block \
	-mllvm --lsr-drop-solution=1 \
	-mllvm -enable-post-misched=0 \
	-mllvm -amdgpu-coerce-illegal-types=1 \
	-mllvm -amdgpu-early-inline-all=true \
	-mllvm -amdgpu-function-calls=false \
	-mllvm -greedy-reverse-local-assignment=1 \
	-I../../include

dbg_cflags+=-g -O0 -std=gnu++20 -Wno-unused-value \
	-I../../include

timestamp=$(shell date +'%Y%m%d_%H%M%S')

# profiler utils
stt_opt=--pmc "LdsBankConflict \
	LdsLatency VALUUtilization" \
	--output-format csv

att_opt=--att-perfcounter-ctrl 10 \
	--att-perfcounters "SQ_BUSY_CU_CYCLES \
		SQ_VALU_MFMA_BUSY_CYCLES \
		SQ_ACTIVE_INST_VALU \
		SQ_ACTIVE_INST_LDS \
		SQ_ACTIVE_INST_VMEM \
		SQ_ACTIVE_INST_FLAT \
		SQ_ACTIVE_INST_SCA \
		SQ_ACTIVE_INST_MISC"

# build flow
% : %.cpp
	hipcc ${cflags} $< -o $*.out -lutility

%.dbg : %.cpp
	hipcc -g ${dbg_cflags} $< -o $*.out -lutility

%.fast : %.cpp
	hipcc ${dbg_cflags} $< -o $*.out -lutility

%.obj : %.cpp
	hipcc ${cflags} --save-temps -Wno-gnu-line-marker $< -o $*.out -lutility

clean:
	rm -rf *.out *out*

clean-tmp:
	rm -rf *.s *.bc *.hipi *.hipfb *.o *-amdgcn-* *gpucore.* .rocprofv3

# simple test cases
test_0 = -F 1 -n 1 -c 8 --in_h 4 --in_w 4 -k 1 -y 3 -x 3 \
--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2
test_1 = -F 1 -n 2 -c 8 --in_h 64 --in_w 64 -k 8 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2
test_2 = -F 1 -n 256 -c 8 --in_h 64 --in_w 64 -k 8 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2
test_3 = -F 1 -n 256 -c 256 --in_h 56 --in_w 56 -k 256 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 32 -m conv --spatial_dim 2
test_4 = -F 1 -n 1024 -c 128 --in_h 28 --in_w 28 -k 32 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2
test_5 = -F 1 -n 1 -c 16 --in_h 960 --in_w 540 -k 16 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1

run_%:
	./igemm.out ${$*} > $*.out 2>&1

st_%:
	rocprofv3 ${stt_opt} -d st_$*_${timestamp}.out -- ./igemm.out ${$*}

tt_%:
	rocprofv3 --att -d tt_$*_${timestamp}.out ${att_opt} -- ./igemm.out ${$*}


# mio ref flow
mio_cmd = MIOpenDriver convfp16 \
	-F 1 -n 256 -c 256 --in_h 56 --in_w 56 -k 256 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 32 -m conv \
	--spatial_dim 2 -t 1 -i 5 # win 53 igem 137

run_mio_cmd:
	${mio_cmd}

mio_st_%:
	rocprofv3 ${stt_opt} -d mio_st_${timestamp}.out -- MIOpenDriver convfp16 ${$*}

mio_tt_%:
	rocprofv3 --att -d mio_tt_${timestamp}.out ${att_opt} -- MIOpenDriver convfp16 ${$*}
# 	rocprofv3 --att -d ttrace_mio_${timestamp}.out ${att_opt} --kernel-include-regex .*conv.* -- ${mio_cmd}


# ck profiler setting
# conv common | op_name         | dtype | layout | index | verify | init | print | time |
conv_common =   grouped_conv_fwd  1       2        0       1        2      0       1

# shape | common        | dim | G  | N   | K/G | C/G | f_len | i_len   | f_stride | f_dilation | i_lpads | i_rpads |
conv_0 =  ${conv_common}  2     4    16    56    56    3 3     235 368   1 1        1 1          1 1       1 1
conv_1 =  ${conv_common}  2     1    1024  32    128   3 3     28 28     1 1        1 1          1 1       1 1
conv_2 =  ${conv_common}  2     32   256   8     8     3 3     56 56     1 1        1 1          1 1       1 1
conv_3 =  ${conv_common}  2     1    1     16    16    3 3     960 450   1 1        1 1          1 1       1 1

# config   | op_name         | dtype | layout | verify | init | print | time | M     | N     | K   | StrideA | StrideB | StrideC | SplitK |
gemm_0   =   gemm_universal    1       0        1        2      0       1      16      518400  144   0         0         0         0
gemm_1   =   gemm_universal    1       1        1        2      0       1      16      518400  144   0         0         0         0
gemm_2   =   gemm_universal    1       0        1        2      0       1      518400  16      144   0         0         0         0 
gemm_3   =   gemm_universal    1       0        1        2      0       1      32      802816  1152  0         0         0         0
gemm_4   =   gemm_universal    1       1        1        2      0       1      32      802816  1152  0         0         0         0

ckprof_%:
	../../build/bin/ckProfiler ${$*} > ckprof_$*.out 2>&1