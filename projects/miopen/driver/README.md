# MIOpenDriver

The `MIOpenDriver` enables the user to test the functionality of any particular
layer in MIOpen in both the forward and backward direction. MIOpen is shipped with `MIOpenDriver` and its install directory is `miopen/bin` located in the install directory path.


## Building the Driver

MIOpenDriver can be build by typing:

```make MIOpenDriver``` from the ```build``` directory.


## Base Arguments
All the supported layers in MIOpen can be found by the supported `base_args` here:

``` ./bin/MIOpenDriver --help ```

The supported base arguments:

 * `conv` - Convolutions
 * `CBAInfer` - Convolution+Bias+Activation fusions for inference
 * `CAInfer` - Convolution+Activation fusions for inference
 * `pool` - Pooling
 * `lrn` - Local Response Normalization
 * `activ` - Activations
 * `softmax` - Softmax
 * `bnorm` - Batch Normalization
 * `rnn` - Recurrent Neural Networks (including LSTM and GRU)
 * `gemm` - General Matrix Multiplication
 * `ctc` - CTC Loss Function
 * `dropout` - Dropout
 * `tensorop` - Ternary Tensor Operation
 * `reduce` - Reduce
 * `layernorm` - Layer Normalization
 * `groupnorm` - Group Normalization
 * `cat` - Cat Forward Operation
 * `addlayernorm` - Add and Layer Normalization
 * `t5layernorm` - T5 Layer Normalization
 * `adam` - Adam Optimizer
 * `ampadam` - AMP Adam Optimizer
 * `adamw` - AdamW Optimizer
 * `ampadamw` - AMP AdamW Optimizer
 * `transformersadamw` - Hugging Face Transformer AdamW Optimizer
 * `transformersampaadamw` - Hugging Face Transformer AMP AdamW Optimizer
 * `getitem` - Getitem Operation
 * `reducecalculation` - Reduce Calculation
 * `rope` - Rotary Position Embedding
 * `prelu` - Parametric ReLU
 * `kthvalue` - Kthvalue Operation
 * `glu` - Gated Linear Unit
 * `softmarginloss` - Softmarginloss
 * `multimarginloss` - Multimarginloss

 These base arguments support fp32 float type, but some of the drivers suport further datatypes -- specifically, half precision (fp16), brain float16 (bfp16), and 8-bit integers (int8).
 To toggle half precision simpily add the suffix `fp16` to end of the base argument; e.g., `convfp16`.
 Likewise, to toggle brain float16 just add the suffix `bfp16`, and to use 8-bit integers add `int8`.

 Notes for this release:
  * Only convolutions support int8
  * Only reduce supports double-precision fp64
  * RNN's support fp16 but only on the HIP backend
  * CTC loss function only supports fp32

Summary of base_args meant for different datatypes and different operations:

| base_args            | Single-Precision (fp32) | Half-Precision (fp16) | Bfloat16 (bfp16)   |
| :------------------- | :---------------------: | :-------------------: | :----------------: |
| conv                 | ✓ | ✓ | ✓ |
| CBAInfer             | x | x | ✓ |
| CAInfer              | x | x | ✓ |
| pool                 | ✓ | ✓ | x |
| lrn                  | ✓ | ✓ | x |
| activ                | ✓ | ✓ | x |
| softmax              | ✓ | ✓ | x |
| bnorm                | ✓ | ✓ | ✓ |
| rnn                  | ✓ | ✓ | x |
| gemm                 | ✓ | ✓ | x |
| ctc                  | ✓ | x | x |
| dropout              | ✓ | ✓ | x |
| tensorop             | ✓ | x | x |
| reduce               | ✓ | ✓ | x |
| layernorm            | ✓ | ✓ | ✓ |
| groupnorm            | ✓ | ✓ | ✓ |
| cat                  | ✓ | ✓ | ✓ |
| addlayernorm         | ✓ | ✓ | ✓ |
| t5layernorm          | ✓ | ✓ | ✓ |
| adam                 | ✓ | ✓ | x |
| ampadam              | ✓ | x | x |
| reduceextreme        | ✓ | ✓ | ✓ |
| adamw                | ✓ | ✓ | x |
| ampadamw             | ✓ | x | x |
| transformersadamw    | ✓ | ✓ | x |
| transformersampadamw | ✓ | x | x  |
| getitem              | ✓ | ✓ | ✓ |
| reducecalculation    | ✓ | ✓ | ✓ |
| rope                 | ✓ | ✓ | ✓ |
| prelu                | ✓ | ✓ | ✓ |
| kthvalue             | ✓ | ✓ | ✓ |
| glu                  | ✓ | ✓ | ✓ |
| softmarginloss       | ✓ | ✓ | ✓ |
| multimarginloss      | ✓ | ✓ | ✓ |

## Executing MIOpenDriver

To execute from the build directory:

```./bin/MIOpenDriver *base_arg* *layer_specific_args*```

Or to execute the default configuration simpily run:

```./bin/MIOpenDriver *base_arg*```

MIOpenDriver example usages:

- Convolution with search on:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2```

- Forward convolution with search off:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```

- Convolution with half or bfloat16 input type

```./bin/MIOpenDriver convfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```
```./bin/MIOpenDriver convbfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```

- Pooling with default parameters:

```./bin/MIOpenDriver pool```

- LRN with default parameters and timing on:

```./bin/MIOpenDriver lrn -t 1```

- Batch normalization with spatial fwd train, saving mean and variance tensors:

```./bin/MIOpenDriver bnorm -F 1 -n 32 -c 512 -H 16 -W 16 -m 1 -s 1```

- RNN with forward and backwards pass, no bias, bi-directional and LSTM mode

```./bin/MIOpenDriver rnn -n 4,4,4,3,3,3,2,2,2,1 -k 10 -H 512 -W 1024 -l 3 -F 0 -b 0 -r 1 -m lstm```

- Printout layer specific input arguments:

`./bin/MIOpenDriver *base_arg* -?` **OR**  `./bin/MIOpenDriver *base_arg* -h (--help)`

Note: By default the CPU verification is turned on. Verification can be disabled using `-V 0`.

## Environment Variables

### Kernel Name and Execution Time Logging

The `MIOPEN_PERFORMANCE_LOGS` environment variable enables lightweight logging of kernel names and their execution times during MIOpenDriver runs. This is useful for debugging, performance analysis, and understanding which kernels are being executed under different configurations (e.g., different `MIOPEN_FIND_MODE` or `MIOPEN_FORCE` settings).

**Logging Levels:**

The variable supports five levels with varying detail and scope:

- **Level 0** (default): No kernel logging
- **Level 1**: Log only **main convolution kernel** per executed solution with **total solution time** - filters out transpose/transform kernels, excludes find/search
- **Level 2**: Log **all kernels** for executed solutions individually - includes transpose/transform kernels, excludes find/search
- **Level 3**: Log only **main convolution kernel** per solution with **total solution time** - filters out transpose/transform kernels, includes find/search
- **Level 4**: Log **all kernels** individually - includes transpose/transform kernels and find/search kernels

**Usage Examples:**

```bash
# Level 1: Log only the chosen/executed kernels
export MIOPEN_PERFORMANCE_LOGS=1
./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2

# Level 2: Log all kernels including find/search
export MIOPEN_PERFORMANCE_LOGS=2
./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2
```

**Output Format:**

The logging uses a hierarchical format where solution names are printed first, followed by the kernels for that solution:

```
[SOLUTION:solution_name]
[KERNEL:exec_id] kernel_name : execution_time ms
[KERNEL:exec_id] kernel_name : execution_time ms
...
```

Where `exec_id` is an execution counter that groups related kernels together (e.g., transpose kernels and their main computation kernel).

**Example Output (Level 1 - main kernels only with total time):**
```
[SOLUTION:ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC]
[KERNEL:1] igemm_fwd_gtcx3_nhwc_bf16... : 1.520 ms (+4 transpose/transform kernels)
[KERNEL:2] igemm_fwd_gtcx3_nhwc_bf16... : 1.489 ms (+4 transpose/transform kernels)
[KERNEL:3] igemm_fwd_gtcx3_nhwc_bf16... : 1.407 ms (+4 transpose/transform kernels)
```

**Example Output (Level 2 - all kernels individually):**
```
[SOLUTION:ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC]
[KERNEL:1] SubTensorOpWithScalar1d : 0.083 ms
[KERNEL:1] batched_transpose_32x32_half : 0.330 ms
[KERNEL:1] batched_transpose_16x32_half : 0.014 ms
[KERNEL:1] igemm_fwd_gtcx3_nhwc_bf16... : 0.893 ms
[KERNEL:1] batched_transpose_32x32_half : 0.269 ms
[KERNEL:2] SubTensorOpWithScalar1d : 0.104 ms
[KERNEL:2] batched_transpose_32x32_half : 0.342 ms
[KERNEL:2] batched_transpose_16x32_half : 0.017 ms
[KERNEL:2] igemm_fwd_gtcx3_nhwc_bf16... : 0.763 ms
[KERNEL:2] batched_transpose_32x32_half : 0.272 ms
```

**Example Output (Level 3 - main kernels only, including find/search):**
```
[SOLUTION:SearchCandidate1]
[KERNEL:1] miopenConv1x1u_search_candidate1 : 1.523 ms
[SOLUTION:SearchCandidate2]
[KERNEL:2] miopenConv1x1u_search_candidate2 : 1.612 ms
[SOLUTION:ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC]
[KERNEL:3] igemm_fwd_gtcx3_nhwc_bf16... : 1.520 ms (+4 transpose/transform kernels)
```

**Example Output (Level 4 - all kernels including find/search):**
```
[SOLUTION:SearchCandidate1]
[KERNEL:1] miopenConv1x1u_search_candidate1 : 1.523 ms
[SOLUTION:SearchCandidate2]
[KERNEL:2] miopenConv1x1u_search_candidate2 : 1.612 ms
[SOLUTION:ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC]
[KERNEL:3] SubTensorOpWithScalar1d : 0.083 ms
[KERNEL:3] batched_transpose_32x32_half : 0.012 ms
[KERNEL:3] igemm_fwd_gtcx3_nhwc_bf16... : 0.234 ms
[KERNEL:3] batched_transpose_32x32_half : 0.013 ms
```

**Key Features:**
- **Multi-level control**: Five levels from no logging to comprehensive kernel-by-kernel logging
- **Noise reduction**: Levels 1 and 3 filter out transpose/transform kernels to focus on core computation
- **Solution-level timing**: Levels 1 and 3 aggregate timing across all kernels in a solution
- **Execution tracking**: `exec_id` groups related kernels that execute together
- **Accurate timing**: Uses GPU events for precise measurement (HIP events for HIP backend, OpenCL profiling for OpenCL backend)
- **Independent of log levels**: Works without changing `MIOPEN_ENABLE_LOGGING`
- **Easy filtering**: Simple `[KERNEL]` prefix for grep/parsing
- **Both backends**: Full support for HIP and OpenCL

**Performance Impact:**
- **Level 0**: No overhead
- **Level 1**: Minimal overhead - only times executed kernels, aggregates at solution level
- **Level 2**: Moderate overhead - times all executed kernels individually
- **Level 3**: Higher overhead - times all solutions including find/search, aggregates at solution level
- **Level 4**: Highest overhead - times every kernel individually including benchmarking runs; synchronizes after each kernel

**JSON Output Mode:**

To enable JSON formatted output, add 256 to the desired log level. JSON mode outputs kernel execution data as structured JSON objects grouped by solution.

```bash
# Level 2 with JSON (2 + 256 = 258)
export MIOPEN_PERFORMANCE_LOGS=258
./bin/MIOpenDriver convbfp16 -W 1024 -H 1024 -c 128 -k 128 -x 3 -y 3
```

**JSON Output Format:**

Each solution outputs a single JSON object containing all its kernels:

```json
{
  "solution": "miopenConvolutionFwdAlgoImplicitGEMM",
  "phase": "execution",
  "kernels": [
    {
      "exec_id": 1,
      "kernel_name": "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt128x128x32...",
      "time_ms": 1.60839,
      "timestamp": "2026-02-04T09:32:39.123456",
      "is_transform": false
    },
    {
      "exec_id": 1,
      "kernel_name": "batched_transpose_32x32_half",
      "time_ms": 0.234,
      "timestamp": "2026-02-04T09:32:39.125012",
      "is_transform": true
    }
  ]
}
```

**JSON Fields:**
- `solution`: Name of the solver/algorithm
- `phase`: Either "execution" (actual computation) or "tuning" (find/search phase)
- `kernels`: Array of kernel execution records with:
  - `exec_id`: Execution counter grouping related kernels
  - `kernel_name`: Full kernel name
  - `time_ms`: Execution time in milliseconds
  - `timestamp`: ISO 8601 timestamp with microsecond precision
  - `is_transform`: Boolean indicating if kernel is transpose/transform operation

**Parsing JSON Output:**
```bash
# Extract kernel times with jq
export MIOPEN_PERFORMANCE_LOGS=258
./bin/MIOpenDriver conv ... 2>&1 | grep "^{" | jq '.kernels[].time_ms'

# Get solution names
./bin/MIOpenDriver conv ... 2>&1 | grep "^{" | jq -r '.solution'

# Filter by phase
./bin/MIOpenDriver conv ... 2>&1 | grep "^{" | jq 'select(.phase == "execution")'
```

**Filtering Output (Traditional Format):**
```bash
# Show only kernel logs
./bin/MIOpenDriver conv ... 2>&1 | grep "\[KERNEL\]"

# Extract just kernel names
./bin/MIOpenDriver conv ... 2>&1 | grep "\[KERNEL\]" | awk '{print $2}'

# Extract kernel names and times
./bin/MIOpenDriver conv ... 2>&1 | grep "\[KERNEL\]" | awk '{print $2, $4, $5}'
```
