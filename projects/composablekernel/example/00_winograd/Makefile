exec_name=winograd

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
test_0 = -F 1 -n 1 -c 12 --in_h 8 --in_w 32 -k 64 -y 3 -x 3 \
--pad_h 1 --pad_w 1 -u 1 -v 1 -l 1 -j 1 -g 1 -m conv --spatial_dim 2

run_%:
	./${exec_name}.out ${$*} > $*.out 2>&1

st_%:
	rocprofv3 ${stt_opt} -d st_$*_${timestamp}.out -- ./${exec_name}.out ${$*}

tt_%:
	rocprofv3 --att -d tt_$*_${timestamp}.out ${att_opt} -- ./${exec_name}.out ${$*}