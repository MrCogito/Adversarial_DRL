#!/bin/sh
mkdir -p outputs/1Random/Markdown
bsub -o "outputs/1Random/Markdown/1Random_0.md" -J "1Random_0" -env MYARGS="-name 1Random-0 -time 84600 -epochs 10000 -batch_size 32 -isServer True -gamma 99.0 -random True -ID 0" < submit_gpu_a80.sh
mkdir -p outputs/1SP/Markdown
bsub -o "outputs/1SP/Markdown/1SP_0.md" -J "1SP_0" -env MYARGS="-name 1SP-0 -time 84600 -epochs 10000 -batch_size 32 -isServer True -gamma 99.0 -random False -ID 0" < submit_gpu_a80.sh

# mkdir -p outputs/1Random/Markdown
# bsub -o "outputs/1Random/Markdown/1Random_0.md" -J "1Random_0" -env MYARGS="-name 1Random-0 -time 84600 -epochs 10000 -batch_size 32 -isServer True -gamma 99.0 -random True -ID 0" < submit_gpu_a80.sh
# #!/bin/sh
# mkdir -p outputs/1SP/Markdown
# bsub -o "outputs/1SP/Markdown/1SP_0.md" -J "1SP_0" -env MYARGS="-name 1SP-0 -time 84600 -epochs 10000 -batch_size 32 -isServer True -gamma 99.0 -random False -ID 0" < submit_gpu_a80.sh