# Qwen VL - Docker Build

For inference and fine tuning.

## Usage

Note, this multi modal model requires GPUs to run with 10gb to 13Gb of VRAM for the Int4 quantized variant used by default.

### Explain an image passed by URl using the default prompt

```
$ docker run --rm --gpus all rahulpowar/qwen-vl ./qwen-demo.py \
    "https://www.looper.com/img/gallery/heres-who-played-darth-vader-without-his-helmet/intro-1566225818.jpg" \
    "https://assets.editorial.aetnd.com/uploads/2014/03/gettyimages-1222666416.jpg"
```

```
...
Loading checkpoint shards: 100%|██████████| 5/5 [00:01<00:00,  3.36it/s]
This is the image of Darth Vader with blood on his head and face, staring at something in a dark room.
This is a photo of the Eiffel Tower at night, as seen from the Trocadero in Paris. The tower is illuminated in gold, and a spotlight shines towards the ground.
```

### Custom prompt and LoRA called checkpoint-5000

```
$ docker run --rm --gpus all -v ./checkpoint-5000:/model/finetuned  rahulpowar/qwen-vl ./qwen-demo.py --lora-model "/model/finetuned" --prompt "\nDescribe Picture 1." \
    "https://www.looper.com/img/gallery/heres-who-played-darth-vader-without-his-helmet/intro-1566225818.jpg" \
    "https://assets.editorial.aetnd.com/uploads/2014/03/gettyimages-1222666416.jpg"
```

## FAQ

Q: I get the following.
```
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: mount error: failed to add device rules: unable to find any existing device filters attached to the cgroup: bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted: unknown.
```
A: This version of the container runtime may need to run as root to access the GPU. `sudo docker run ...`

Q: How do I setup docker on Windows to support GPUs?
A: https://medium.com/htc-research-engineering-blog/nvidia-docker-on-wsl2-f891dfe34ab

Q: How do I check the GPU is working correctly in docker?
A: 
```
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark

...
> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
MapSMtoArchName for SM 8.9 is undefined.  Default to use Ampere
GPU Device 0: "Ampere" with compute capability 8.9

> Compute 8.9 CUDA device: [NVIDIA GeForce RTX 4090]
131072 bodies, total time for 10 iterations: 77.294 ms
= 2222.671 billion interactions per second
= 44453.425 single-precision GFLOP/s at 20 flops per interaction
```

Q: I sometimes get mandarin! 这是什么鬼?
A: Yes, the model is from Alibaba Cloud and has a broad training set. However as it is OSS, many approaches from token bias to fine tuning will work.