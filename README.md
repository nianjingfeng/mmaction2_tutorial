# PoseC3D Tutorial

## Enviroment
- Linux Ubuntu 22.04
- GPU RTX 3070
- NVIDIA Driver 530.30.02
- CUDA Version 12.0
- Docker Version 23.0.2
- NVIDIA Container Toolkit Version 1.13.0
## Preparation
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA CONTAINER TOOLKIT](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Visual Studio Code IDE
- Git
- [Training data](https://drive.google.com/file/d/1qmcXhWKceV1nEFOeTj1JyNEGcSPkO7pV/view?usp=share_link), you can check the structure of training file by this. 
## Step
---
1. Download the repo.
```
git clone https://github.com/nianjingfeng/mmaction2_tutorial.git
cd mmaction2_tutorial
```
3. Build the docker image.
```
docker build -t mmaction2 .
```
2. Build the docker container.
Shared memory size can be modified denpends on source of server.
```
docker run -it --gpus all --shm-size 16G -u root -v .:/mmaction2/data mmaction2 /bin/bash
```
3. Install the requirement library.
```
pip install -v -e .
```
4. Verification by test file.
```
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
mv demo/demo.py .
python demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```
```
The output is shown below:
```

![result](https://github.com/nianjingfeng/mmaction2_tutorial/blob/master/line_20230508_195041.png)

5. Set config and training file to proper path.
```
mv data/5g_train.py /mmaction2/tools/5g_train.py
mv data/5g_config.py /mmaction2/configs/skeleton/posec3d/5g_config.py
```
6. Set up setting in config file.
- ann_file = training file path under docker
- num_classes = number of classes in training data
- batch_sizr = number of training data each iteration
- max_epochs = number of training epochs
7. Start training in IDE.
## Citation
---
[mmaction2](https://github.com/open-mmlab/mmaction2)