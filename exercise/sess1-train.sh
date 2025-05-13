#!/bin/bash
abspath=$(cd "$(dirname "$0")"; pwd)

:<<"eof"
trl sft --model_name_or_path HuggingFaceTB/SmolLM2-135M \
    --dataset_name HuggingFaceTB/smoltalk \
    --dataset_config everyday-conversations \
    --output_dir $abspath/junhangc
===>
This simulates running: `accelerate launch <launch args> sft.py <training script args>`.
eof

accelerate launch exercise/sess1-sft.py