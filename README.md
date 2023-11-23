# Qwen VL - Docker Build

For inference and fine tuning.

## Usage

Note, this multi modal requires GPUs to run with about 13Gb of VRAM for the Int4 quantized variant used by default.

## Explain an image passed by URl using the default prompt

```
$ docker run --rm --gpus all rahulpowar/qwen-vl ./qwen-demo.py \
    "https://www.looper.com/img/gallery/heres-who-played-darth-vader-without-his-helmet/intro-1566225818.jpg" \
    "https://cdn.britannica.com/54/75854-050-E27E66C0/Eiffel-Tower-Paris.jpg"
```

```
...
Loading checkpoint shards: 100%|██████████| 5/5 [00:01<00:00,  3.36it/s]
This is the image of Darth Vader with blood on his head and face, staring at something in a dark room.
This is a photo of the Eiffel Tower at night, as seen from the Trocadero in Paris. The tower is illuminated in gold, and a spotlight shines towards the ground.
```

## FAQ

Q: I get the following.

```
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: mount error: failed to add device rules: unable to find any existing device filters attached to the cgroup: bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted: unknown.
```

A: This version of the container runtime may need to run as root to access the GPU.

Q: How do I setup docker on Linux to support GPUs?

A:

Q: How do I setup docker on Windows to support GPUs?

A:
