# Tracking Everything in Robotic-Assisted Surgery

## Installation
The code is tested with `python=3.8` and `torch=1.10.0+cu111`

```
git clone --recurse-submodules https://github.com/zhanbh1019/SurgicalMotion/
cd SurgicalMotion/
conda create -n surgicalmotion python=3.8
conda activate surgicalmotion
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib tensorboard scipy opencv-python tqdm tensorboardX configargparse ipdb kornia imageio[ffmpeg]
```

## Training
1. Please refer to the [preprocessing instructions](preprocessing/README.md) for preparing input data for training SurgicalMotion.
2. With processed input data, run the following command to start training:
    ```
    python train.py --config configs/default.txt --data_dir {sequence_directory}
    ```
    You can view visualizations on tensorboard by running `tensorboard --logdir logs/`. 
