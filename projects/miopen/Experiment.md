I need you to figure out why the HIP version is much slower than the OCL version of this kernel.

The HIP kernel can be found at /home/AMD/dahawkin/work/ck_study/worktrees/ocl-hip-kernel-conversion/projects/miopen/src/kernels/MIOpenGroupConvBwdWrW_LxG_P53Hip.cpp
The OCL kernel can be found at /home/AMD/dahawkin/work/ck_study/worktrees/ocl-hip-kernel-conversion/projects/miopen/src/kernels/MIOpenGroupConvBwdWrW_LxG_P53.cl

If you want to run the experiment you can run rerun-exp.sh from the miopen project directory. This will rebuild the project, clear logs from WIP/perf, and clear the kernel database. Then, it will run the performance tests for HIP and OCL. Note: These kernels are runtime compiled, so any actual kernel compilation failures won't manifest until then. The output will tell you the perf result at the end of running it. That line looks like "Speedup mean(OCL)/mean(HIP): 0.74x (>1 means HIP faster)". If you don't see this in the output, it likely means the kernel failed to compile at runtime. To find the kernel compilation error message uou can look at the deeper logs from the HIP perf run in the file(s) WIP/perf/cfg6-*-*-run*-hip.log. These get cleared and assigned a slightly new name every time you run the experiment so you will need to find the current set after the experiment runs.

The compiled kernel code can be found in a kernel database at a path of the format /tmp/.cache/miopen/*/*.kdb. This kernel database contains bzip2'd compiled kernel binaries. Both the OpenCL and HIP kernels will be in this kdb, but the OpenCL kernel's compiled code should not be changing from experiment to experiment.

# Structure

Approach this by having a researcher agent look at the implementations to come up with experiments to try to improve performance. Then, have an experiment agent make the changes and run the experiment. It's very important you do it in this manner so that you don't prematurely deplete your own context window. You are the experiment coordinator.

You will be managing researcher and experimenter agents. Create a new researcher agent for each research task and a new experimenter agent for each experiment task attempt. Their personas are later in this document. When your researcher agent comes up with new experiment tasks, create a new experimenter agent to implement and execute the experiment. When there are results from the experiment, create a new researcher agent to examine the results (including any updated generated AMD GCN ISA code to compare against the generated OCL ISA code) and determine it potentially generates any new experiments, and add them to the todo list.

When managing your agents, do NOT transitively give them the role of experiment coordinator. This persona cannot be transferred.

Only run one experiment at a time.

Keep a to-do list of experiments that you plan to run and ensure they are visible to me in your main window.

Success for this is to achieve an OCL vs HIP performance ratio of over 0.85x. That is, the HIP kernel is at least 85% as performant as the OCL kernel. When you have reached this, ensure you test correctness of your changes. You can do this by running the WIP/verify_convoclbwdwrw53_gtest.sh and WIP/verify_convoclbwdwrw53.sh scripts. The correctness constraints outlined in the tests in this project cannot be changed. Do not change them to loosen the constraints of correctness. If you reach the performance requirement but the code is now incorrect, you need to find another avenue for improvements.

Once you meet the performance and correctness requirements, stop your experiments and write the results to a markdown file within the WIP directory. Stage and commit any code changes you have made with a sensible commit message but do not push them.

# Researcher Persona

You are an AMD GPU convolutional kernel performance researcher. You are able to understand HIP code, OpenCL code, as well as AMD GCN ISA code. You will likely need to examine and compare HIP code with OpenCL code, as well as AMD GCN ISA code from a runtime compiled HIP kernel.

# Experimenter Persona

The experimenter should run the experiment script given above and pass the results to the researcher persona. If there are compile errors or HIP kernel runtime compile errors, the experiment needs to be iterated on to get the kernel to compile. Instructions for determining the compile error are earlier in this document. If there is no way to get this kernel to compile, report that back to the researcher agent and see if they have a different variation for you to try.