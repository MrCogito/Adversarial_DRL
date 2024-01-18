#!/bin/sh
mkdir -p outputs/2024v32_notdone2/Markdown
bsub -o "outputs/2024v32_notdone2/Markdown/2024v32_notdone2_0.md" -J "2024v32_notdone2_0" -env MYARGS="-name 2024v32_notdone2-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_v32.sh
mkdir -p outputs/2024a80_notdone2/Markdown
bsub -o "outputs/2024a80_notdone2/Markdown/2024a80_notdone2_0.md" -J "2024a80_notdone2_0" -env MYARGS="-name 2024a80_notdone2-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.003 -ID 0" < submit_gpu_a80.sh
mkdir -p outputs/2024v32_notdone3/Markdown
bsub -o "outputs/2024v32_notdone3/Markdown/2024v32_notdone3_0.md" -J "2024v32_notdone3_0" -env MYARGS="-name 2024v32_notdone3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.00015 -ID 0" < submit_gpu_v32.sh
mkdir -p outputs/2024a80_notdone3/Markdown
bsub -o "outputs/2024a80_notdone3/Markdown/2024a80_notdone3_0.md" -J "2024a80_notdone3_0" -env MYARGS="-name 2024a80_notdone3-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -lr 0.00015 -ID 0" < submit_gpu_a80.sh
