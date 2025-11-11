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
data_type = 1
indexing = 0
verify = 1
init = 2
print = 0
time = 1

dim = 2
G = 32
N = 256
K = $(shell expr 256 / ${G})
C = $(shell expr 256 / ${G})

filter_lengths = 3 3
input_lengths = 56 56
filter_strides = 1 1
filter_dilations = 1 1
input_left_pads = 1 1
input_right_pads = 1 1

# config
# layout = 0
# layout = 1
layout = 2

run_ck_profiler:
	../../build/bin/ckProfiler grouped_conv_fwd ${data_type} ${layout} ${indexing} \
		${verify} ${init} ${print} ${time} ${dim} ${G} ${N} ${K} ${C} ${filter_lengths} \
		${input_lengths} ${filter_strides} ${filter_dilations} ${input_left_pads} \
		${input_right_pads} > layout_${layout}.log