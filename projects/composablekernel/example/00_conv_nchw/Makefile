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
att_opt=--att-perfcounter-ctrl 10 \
	--att-perfcounters "SQ_BUSY_CU_CYCLES \
		SQ_VALU_MFMA_BUSY_CYCLES \
		SQ_ACTIVE_INST_VALU \
		SQ_ACTIVE_INST_LDS \
		SQ_ACTIVE_INST_VMEM \
		SQ_ACTIVE_INST_FLAT \
		SQ_ACTIVE_INST_SCA \
		SQ_ACTIVE_INST_MISC"

prof_cfg=-F 2 -n 24 -c 256 --in_h 5 --in_w 5 \
	-k 510 -y 3 -x 3 --pad_h 1 --pad_w 1 \
	-u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1

mio_cmd = MIOpenDriver convfp16 \
	-F 1 -n 256 -c 256 --in_h 56 --in_w 56 -k 256 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 32 -m conv \
	--spatial_dim 2 -t 1 -i 5 # win 53 igem 137

prof_name=igemm

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
test_4 = -F 1 -n 1024 -c 128 --in_h 56 --in_w 56 -k 32 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2
test_5 = -F 1 -n 1 -c 16 --in_h 960 --in_w 540 -k 16 -y 3 -x 3 \
	--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2 -t 1

run_%:
	./igemm.out ${$*} > $*.out 2>&1

prof_%:
	rocprofv3 --att -d ttrace_$*_${timestamp}.out ${att_opt} -- ./igemm.out ${$*}

# mio ref flow
run_mio_cmd:
	${mio_cmd}

prof_mio_strace:
	rocprofv3 --sys-trace --output-format pftrace -d strace_${prof_name}_${timestamp}.out -- ${mio_cmd}

prof_mio_ttrace:
	rocprofv3 --att -d ttrace_${prof_name}_${timestamp}.out ${att_opt} -- ${mio_cmd}
# 	rocprofv3 --att -d ttrace_${prof_name}_${timestamp}.out ${att_opt} --kernel-include-regex .*conv.* -- ${mio_cmd}

# ck profiler setting
# kernel config   | op_name         | dtype | layout | index | verify | init | print | time |
kernel_config_0 =   grouped_conv_fwd  1       2        0       1        2      0       1

# shape          | dim | G  | N   | K/G | C/G | f_len | i_len   | f_stride | f_dilation | i_lpads | i_rpads |
shape_config_0 =   2     4    16    56    56    3 3     235 368   1 1        1 1          1 1       1 1
shape_config_1 =   2     1    1024  32    128   3 3     28 28     1 1        1 1          1 1       1 1
shape_config_2 =   2     32   256   8     8     3 3     56 56     1 1        1 1          1 1       1 1
shape_config_3 =   2     1    1     16    16    3 3     960 450   1 1        1 1          1 1       1 1

run_ckprof:
# 	../../build/bin/ckProfiler ${kernel_config_0} ${shape_config_0} > ckprof_0.out 2>&1
# 	../../build/bin/ckProfiler ${kernel_config_0} ${shape_config_1} > ckprof_1.out 2>&1
# 	../../build/bin/ckProfiler ${kernel_config_0} ${shape_config_2} > ckprof_2.out 2>&1
	../../build/bin/ckProfiler ${kernel_config_0} ${shape_config_3} > ckprof_3.out 2>&1

# kernel config   | op_name         | dtype | layout | verify | init | print | time | M  | N     | K   | StrideA | StrideB | StrideC | SplitK |
gemm_config_0   =   gemm_universal    1       0        1        2      0       1      16   518400  144   144       518400    518400    0
gemm_config_1   =   gemm_universal    1       0        1        2      0       1      32   802816  1152  1152      802816    802816    0
gemm_config_2   =   gemm_universal    1       1        1        2      0       1      16   518400  144   144       144       518400    0

run_ckprof_g:
# 	../../build/bin/ckProfiler ${gemm_config_0} > ckprof_g_0.out 2>&1
	../../build/bin/ckProfiler ${gemm_config_1} > ckprof_g_1.out 2>&1