# PyTorch Oracle

`dsv4_pytorch_oracle.py` is the correctness baseline for fixed-batch logprob
comparisons. It uses the exact reduced DSV4 dimensions from
`../config/dsv4_reduced_12l.json`, deterministic virtual weights, and a fixed
token batch. The output `.pt` file contains:

- `tokens`: fixed input tokens
- `logprobs`: next-token logprobs with shape `[batch, seq_len - 1]`
- `config`: shape and seed metadata
- `elapsed_sec`: wall-clock runtime for the oracle pass

Small local syntax/smoke runs can override `--seq-len` and `--layers`; measured
comparisons should use the defaults.

```bash
python references/amd/dsv4-pretrain-megatron/oracle/dsv4_pytorch_oracle.py \
  --output /tmp/dsv4_oracle.pt \
  --metrics-json /tmp/dsv4_oracle.json
```
