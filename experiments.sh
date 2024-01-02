#!/bin/sh
mkdir -p outputs/2024v32_newscoresRand0/Markdown
bsub -o "outputs/2024v32_newscoresRand0/Markdown/2024v32_newscoresRand0_0.md" -J "2024v32_newscoresRand0_0" -env MYARGS="-name 2024v32_newscoresRand0-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -random True -ID 0" < submit_gpu_v32.sh
mkdir -p outputs/2024v80_nrescoresRand0/Markdown
bsub -o "outputs/2024v80_nrescoresRand0/Markdown/2024v80_nrescoresRand0_0.md" -J "2024v80_nrescoresRand0_0" -env MYARGS="-name 2024v80_nrescoresRand0-0 -time 84600 -epochs 1000 -batch_size 32 -isServer True -gamma 99.0 -random True -ID 0" < submit_gpu_a80.sh
