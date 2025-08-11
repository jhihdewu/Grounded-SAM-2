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
