```
Problem                                     Direction      Time     Error    Cmd
──────────────────────────────────────────────────────────────────────────────────
3×3, g=32, c=k=128, 200×200 (4ch/group)    Fprop          0.243ms  2.04e-06  [1]
3×3, g=32, c=k=128, 200×200 (4ch/group)    Backward Data  0.245ms  2.00e-06  [2]
3×3, g=32, c=k=512, 50×50  (16ch/group)    Fprop          0.041ms  2.78e-06  [3]
3×3, g=32, c=k=512, 50×50  (16ch/group)    Backward Data  0.040ms  2.68e-06  [4]

[1] MIOpenDriver convfp16 -n 32 -c 128 -H 200 -W 200 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 1 -t 1
[2] MIOpenDriver convfp16 -n 32 -c 128 -H 200 -W 200 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 2 -t 1
[3] MIOpenDriver convfp16 -n 32 -c 512 -H 50 -W 50 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 1 -t 1
[4] MIOpenDriver convfp16 -n 32 -c 512 -H 50 -W 50 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -m conv -g 32 -F 2 -t 1
```
