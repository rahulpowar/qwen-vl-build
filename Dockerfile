FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt install python3 python3-pip python-is-python3 libopenmpi-dev -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# relevant code is in the models but required for fine tuning
ADD qwen-vl /app/qwen-vl

ADD qwen-demo.py /app

# Install dependencies
# Funky set of specific versions required to allow both inference and training of Int4 LoRRAs
RUN cd qwen-vl && python3 -m pip install --no-cache-dir -r requirements.txt hf_transfer mpi4py deepspeed wandb peft==0.6.0 optimum==1.13.2 auto-gptq==0.4.2

# https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download the Int4 model for use
RUN huggingface-cli download Qwen/Qwen-VL-Chat-Int4 
RUN ./qwen-demo.py

VOLUME [ "/app/qwen-vl/output_qwen" ]