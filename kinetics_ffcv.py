from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBVideoDecoder
from ffcv.fields import RGBVideoField
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, Squeeze, ToDevice, \
    ToTorchVideo, NormalizeVideo, VideoRandomHorizontalFlip

import torch
import torchvision

import time
from typing import List
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


if __name__ == '__main__':
    data_path = '/scratch/jeh16/ffcv/k400/k400_train_16x4.ffcv'
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

    num_samples = 2261765
    epochs = 10 # simulate 10 epochs of training

    train_decoder = RandomResizedCropRGBVideoDecoder(output_size=224)

    video_pipeline: List[Operation] = [
        train_decoder,
        VideoRandomHorizontalFlip(flip_prob=0.5),
        ToTensor(),
        ToDevice(device=torch.device('cuda:0'), non_blocking=True),
        ToTorchVideo(),
        NormalizeVideo(mean=IMAGENET_MEAN, std=IMAGENET_STD, type=np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]
    vid_index_pipeline : List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]
    clip_index_pipeline : List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]

    pipelines = {
        'video': video_pipeline,
        'label': label_pipeline,
        'video_index': vid_index_pipeline,
        'clip_index': clip_index_pipeline
    }

    order = OrderOption.QUASI_RANDOM
    loader = Loader(
        data_path,
        batch_size=32,
        num_workers=40//2,
        order=order,
        os_cache=True,
        drop_last=True,
        pipelines=pipelines,
        custom_fields={
            'video': RGBVideoField()
        },
        seed=1234
    )
    times = []
    for epoch in range(epochs):
        start = time.time()
        for i, (video, label, _, _) in enumerate(tqdm(loader)):
            pass
        end = time.time()
        times.append(end - start)
        print('epoch', epoch, 'time', end - start)

    print(times)
    print('mean time', np.mean(times), 'std time', np.std(times))
    print('samples per second', num_samples / np.mean(times))