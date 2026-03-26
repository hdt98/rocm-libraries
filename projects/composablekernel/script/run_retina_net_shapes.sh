#!/bin/bash

python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 128 -H 200 -W 200 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 1 -t 0 --instance 21

# Case 2
python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 128 -H 200 -W 200 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 2 -t 0 --instance 10

# Case 3
python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 512 -H 50 -W 50 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 1 -t 0 --instance 126

# Case 4
python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 512 -H 50 -W 50 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 2 -t 0 --instance 29

# Case 5
python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 256 -H 100 -W 100 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 1 -t 0 --instance 127

# Case 6
python3 convert_miopen_driver_to_profiler.py MIOpenDriver convfp16 -n 32 -c 256 -H 100 -W 100 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 2 -t 0 --instance 0