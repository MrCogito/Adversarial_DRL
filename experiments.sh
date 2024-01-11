#!/bin/sh
mkdir -p outputs/2024v32_mar3/Markdown
bsub -o "outputs/2024v32_mar3/Markdown/2024v32_mar3_0.md" -J "2024v32_mar3_0" -env MYARGS="-name 2024v32_mar3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_v32.sh
mkdir -p outputs/2024a80_mar3/Markdown
bsub -o "outputs/2024a80_mar3/Markdown/2024a80_mar3_0.md" -J "2024a80_mar3_0" -env MYARGS="-name 2024a80_mar3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_a80.sh
mkdir -p outputs/2024v16_mar3/Markdown
bsub -o "outputs/2024v16_mar3/Markdown/2024v16_mar3_0.md" -J "2024v16_mar3_0" -env MYARGS="-name 2024v16_mar3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_v16.sh
mkdir -p outputs/2024a40_mar3/Markdown
bsub -o "outputs/2024a40_mar3/Markdown/2024a40_mar3_0.md" -J "2024a40_mar3_0" -env MYARGS="-name 2024a40_mar3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_a40.sh
