#!/bin/sh
# usage: asm-new.sh kernelName(no extension) [--wave32]
f=$1
shift
if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then
    wave=32
    shift
else
    wave=64
fi
h=gfx950
if [ $wave -eq 32 ]; then
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx950 -mno-wavefrontsize64 -c -o $f.o $f.s
else
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx950 -mwavefrontsize64 -c -o $f.o $f.s
fi
/opt/rocm/bin/amdclang++ -target amdgcn-amd-amdhsa -Xlinker --build-id -o $f.co $f.o
cp $f.co ../../../library/${f}_$h.co
mkdir -p ../../../asm_backup && cp $f.s ../../../asm_backup/$f.s
