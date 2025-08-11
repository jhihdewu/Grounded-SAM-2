# Grounded SAM 2: Ground and Track Anything in Videos

This repo is a fork from [IDEA-Research's Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) and is used for generating labels for the take home project.

Main changes I've made are:
1. replace all torch.bfloat16 with torch.float16 as a workaround for this error:
    ```
    RuntimeError: "ms_deform_attn_forward_cuda" not implemented for 'BFloat16'
    ```
2. extend one of the demo example to the new `generate_yolo_dataset.py` script, which is a utility for generating object detection datasets using Grounding DINO and SAM2 models. 

## Contents
- [Installation](#installation)
- [Generate Yolo Dataset](#generate-yolo-dataset)


## Installation

Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

Download the pretrained `Grounding DINO` checkpoints:

```bash
cd gdino_checkpoints
bash download_ckpts.sh
```



### Installation with docker
Note: If you look at the Makefile you will see that whether docker image will be built with CUDA support depends on whether HOST has nvcc or not.
So you need to make sure you have cuda toolkit installed on the "HOST". You cah check this by `nvcc --version`.

Build the Docker image and Run the Docker container:

```
cd Grounded-SAM-2
make build-image
make run
```
After executing these commands, you will be inside the Docker environment. The working directory within the container is set to: `/home/appuser/Grounded-SAM-2`

Once inside the Docker environment, you can start the demo by running:
```
python grounded_sam2_local_demo.py
```
The output will be at `Grounded-SAM-2/outputs/grounded_sam2_local_demo`

#### Trouble Shooting
Note that if you see error like this:
```
python grounded_sam2_local_demo.py
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1716905979055/work/aten/src/ATen/native/TensorShape.cpp:3587.)
final text_encoder_type: bert-base-uncased
tokenizer_config.json: 100%|███████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 153kB/s]
config.json: 100%|██████████████████████████████████████████████████████████| 570/570 [00:00<00:00, 1.10MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 521kB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 888kB/s]
model.safetensors: 100%|██████████████████████████████████████████████████| 440M/440M [01:11<00:00, 6.15MB/s]
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /opt/conda/conda-bld/pytorch_1716905979055/work/torch/csrc/utils/tensor_numpy.cpp:206.)
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
Traceback (most recent call last):
  File "/home/appuser/Grounded-SAM-2/grounded_sam2_local_demo.py", line 58, in <module>
    boxes, confidences, labels = predict(
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/util/inference.py", line 68, in predict
    outputs = model(image[None], captions=[caption])
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/groundingdino.py", line 327, in forward
    hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/transformer.py", line 258, in forward
    memory, memory_text = self.encoder(
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/transformer.py", line 576, in forward
    output = checkpoint.checkpoint(
  File "/opt/conda/lib/python3.10/site-packages/torch/_compile.py", line 24, in inner
    return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 451, in _fn
    return fn(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/_dynamo/external_utils.py", line 36, in inner
    return fn(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py", line 487, in checkpoint
    return CheckpointFunction.apply(function, preserve, *args)
  File "/opt/conda/lib/python3.10/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py", line 262, in forward
    outputs = run_function(*args)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/transformer.py", line 785, in forward
    src2 = self.self_attn(
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/ms_deform_attn.py", line 338, in forward
    output = MultiScaleDeformableAttnFunction.apply(
  File "/opt/conda/lib/python3.10/site-packages/torch/autograd/function.py", line 598, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/ms_deform_attn.py", line 53, in forward
    output = _C.ms_deform_attn_forward(
NameError: name '_C' is not defined

```
Then you need to do these two steps in the docker container before running the demo

```
pip install -e .
pip install --no-build-isolation -e grounding_dino
```

## Generate Yolo Dataset
I've added a script, [generate_yolo_dataset.py](./generate_yolo_dataset.py), which is a utility for generating object detection datasets using Grounding DINO and SAM2 models. It can process a folder of images and create annotations in either JSON format or YOLO segmen.

### Example usage
```
python generate_yolo_dataset.py \
    --image_folder "./my_images" \
    --text_prompt "wok. egg. spoon. bowl." \
    --output_dir "./yolo_dataset" \
    --yolo_output

```

This will label all the images in the my_images folder, and generate an Ultralytics Yolo dataset that is ready for training with Ultralytics API in the yolo_dataset folder.

The text_prompt "wok. egg. spoon. bowl." means our foucs (the label we want) are wok, egg, spoon and bowl. The model will find these things in the image and label it for you if these are found.
