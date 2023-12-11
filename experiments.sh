#!/bin/sh
#mkdir -p outputs/Test40/Markdown
#bsub -o "outputs/Test40/Markdown/Test40_0.md" -J "Test40_0" -env MYARGS="-name Test40-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -ID 0" < submit_gpu_a40.sh
mkdir -p outputs/Test32/Markdown
bsub -o "outputs/Test32/Markdown/Test32_0.md" -J "Test32_0" -env MYARGS="-name Test32-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -ID 0" < submit_gpu_a32.sh
mkdir -p outputs/Test16/Markdown
bsub -o "outputs/Test16/Markdown/Test16_0.md" -J "Test16_0" -env MYARGS="-name Test16-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -ID 0" < submit_gpu_a16.sh
