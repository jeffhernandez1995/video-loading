import os
os.environ['TRANSFORMERS_CACHE'] = '/work/vo9/jeh16/huggingface/'

from ffcv.fields.decoders import IntDecoder, CenterCropRGBVideoDecoder, RandomResizedCropRGBVideoDecoder
from ffcv.fields import RGBVideoField
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, Squeeze, ToDevice, \
    ToTorchVideo, NormalizeVideo

import torch
import torchvision
from torchvision.transforms import functional as F

from transformers import VideoMAEForVideoClassification, \
    TimesformerForVideoClassification, \
    VivitForVideoClassification



import time
from typing import List
import os
from tqdm import tqdm
from PIL import Image
import math
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config
from sklearn.metrics import accuracy_score, top_k_accuracy_score


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


Section('cfg', 'arguments to give the writer').params(
    dataset=Param(
        str,
        'Path to the dataset',
        default='/scratch/jeh16/data/ffcv/k400/k400_val_16x4_1x5.ffcv'
    ),
    model=Param(
        str,
        'Huggingface model name',
        default='MCG-NJU/videomae-small-finetuned-kinetics'
    ),
    num_workers=Param(int, 'Number of workers to use', default=1),
    batch_size=Param(int, 'Batch size', default=24),
    chunk_idx=Param(int, 'Device to use', default=0),
    num_chunks=Param(int, 'Number of chunks', default=8)
)


@section('cfg')
@param('dataset')
@param('model')
@param('num_workers')
@param('batch_size')
@param('chunk_idx')
@param('num_chunks')
def main(
    dataset: str,
    model: str,
    num_workers: int,
    batch_size: int,
    chunk_idx: int,
    num_chunks: int,
) -> None:

    device = torch.device(f'cuda:{chunk_idx}' if torch.cuda.is_available() else 'cpu')

    if 'timesformer-hr' in model:
        output_size = 448
    else:
        output_size = 224
    # train_decoder = CenterCropRGBVideoDecoder(output_size=output_size, ratio=0.875)
    # scale is 1.0, ratio is 1.0 this is just a resize instead of a random crop
    train_decoder = RandomResizedCropRGBVideoDecoder(output_size=output_size, scale=(1.0, 1.0), ratio=(1.0, 1.0))

    if 'videomae' in model:
        IMG_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMG_STD = np.array([0.229, 0.224, 0.225]) * 255
    elif 'timesformer' in model:
        IMG_MEAN = np.array([0.45, 0.45, 0.45]) * 255
        IMG_STD = np.array([0.225, 0.225, 0.225]) * 255
    elif 'vivit' in model:
        IMG_MEAN = np.array([0.5, 0.5, 0.5]) * 255
        IMG_STD = np.array([0.5, 0.5, 0.5]) * 255
    else:
        raise ValueError(f'Unknown model: {model}')

    video_pipeline: List[Operation] = [
        train_decoder,
        ToTensor(),
        ToDevice(device=device, non_blocking=True),
        ToTorchVideo(),
        # NormalizeVideo(mean=IMG_MEAN, std=IMG_STD, type=np.float16)
        NormalizeVideo(mean=IMG_MEAN, std=IMG_STD, type=np.float32) # hugingface models expect float32??
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

    if num_chunks > 0:
        # Hardcoded for kinetics-400
        num_samples = 93780
        num_batches = num_samples // batch_size
        # Drop last
        num_samples = num_batches * batch_size
        indices = list(range(num_batches))
        indices = get_chunk(indices, num_chunks, chunk_idx)
        order = OrderOption.SEQUENTIAL
        loader = Loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=True,
            drop_last=False,
            pipelines=pipelines,
            custom_fields={
                'video': RGBVideoField()
            },
            seed=1234,
            indices=indices,
        )
    else:
        order = OrderOption.SEQUENTIAL
        loader = Loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=True,
            drop_last=True,
            pipelines=pipelines,
            custom_fields={
                'video': RGBVideoField()
            },
            seed=1234,
        )

    # device = torch.device(f'cuda:{chunk_idx}' if torch.cuda.is_available() else 'cpu')
    model_name = model
    if 'videomae' in model:
        model = VideoMAEForVideoClassification.from_pretrained(model_name)
    elif 'timesformer' in model:
        model = TimesformerForVideoClassification.from_pretrained(model_name)
    elif 'vivit' in model:
        model = VivitForVideoClassification.from_pretrained(model_name)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.to(device)

    # transform = torchvision.transforms.Compose([
    #     # BatchRandomHorizontalFlip(),
    #     BatchToTensor(),
    #     BatchNormalize(mean=IMG_MEAN, std=IMG_STD)
    # ])

    true_labels = []
    video_idxs = []
    clip_idxs = []
    predictions = []
    # logits = []
    # k = 5
    with torch.no_grad():
        for i, (videos, labels, video_idx, clip_idx) in enumerate(tqdm(loader)):
            # Video is BxTxHxWxC we need BxTxCxHxW
            # videos = videos.transpose((0, 1, 4, 2, 3))
            if "timesformer-base" in model_name:
                # take 8 frames from the 16
                videos = videos[:, :2, ...]
            if 'vivit' in model_name:
                # repeat the frames to get 32x2 from 16x4
                indices = torch.linspace(0, 15, 32).long()
                videos = videos[:, indices, ...]
            # ffcv already does this
            # videos = transform(videos)
            # videos = videos.to(device)
            true_labels.extend(labels.numpy().tolist())
            video_idxs.extend(video_idx.numpy().tolist())
            clip_idxs.extend(clip_idx.numpy().tolist())
            outputs = model(pixel_values=videos)
            preds = outputs.logits.cpu().numpy().tolist()
            # idx = np.argsort(-preds, axis=1)
            # logs = preds[idx, :k].tolist()
            # preds = idx[:, :k].tolist()
            # logits.extend(logs)
            predictions.extend(preds)

    # conatenate the results along the 0th axis
    data = np.hstack([
        np.array(true_labels)[:, None],
        np.array(video_idxs)[:, None],
        np.array(clip_idxs)[:, None],
        np.array(predictions)
    ])

    logit_cols = [f'logit_{i}' for i in range(400)]
    columns = ['true_label', 'video_idx', 'clip_idx'] + logit_cols
    results = pd.DataFrame(data, columns=columns)
    results.to_csv(f'{num_chunks}_{chunk_idx}.csv', index=False)
    print('Done')

    # group by video index and take the mean of only the logits
    grouped = results.groupby('video_idx').mean()
    # take the argmax of the mean logits
    predictions = grouped[logit_cols].values.argmax(axis=1)
    true_labels = results.groupby('video_idx')['true_label'].first().values
    acc1 = accuracy_score(true_labels, predictions)
    # acc5 = top_k_accuracy_score(true_labels, np.array(predictions), k=5)
    print(f'Accuracy@1: {acc1:.3f}')

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
