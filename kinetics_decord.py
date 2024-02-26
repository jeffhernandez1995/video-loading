import os
import random
import numpy as np
from numpy.lib.stride_tricks import as_strided

import torch
import torch.utils.data

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from itertools import product

from pathlib import Path 

from torch.utils.data import DataLoader
import torchvision
import time
from torch.utils.data.dataloader import default_collate
from PIL import Image

import time
from tqdm import tqdm

import decord as de
from torch.utils.dlpack import to_dlpack, from_dlpack

de.bridge.set_bridge('torch')

# might be more readable like this https://www.kaggle.com/code/ipythonx/pytorch-evaluation-kinetics-400-video-set


class Kinetics(VisionDataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
        self,
        mode,
        data_dir,
        sampling_rate=4,
        num_frames=16,
        temporal_views=7, # -1 for random sampling
        spatial_views=1, # -1 for random sampling
        transform=None,
    ):
        self.mode = mode

        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        if self.mode in ["train"]:
            self.temporal_views = -1 # always random sampling for training
            self.spatial_views = -1 # always random sampling for training
        elif self.mode in ["val", "test"]:
            self.temporal_views = temporal_views
            self.spatial_views = spatial_views
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        assert self.spatial_views in [-1, 1], "Only support spatial_views are random or 1"
        self.transform = transform

        classes = list(sorted(list_dir(str(self.data_dir / mode))))
        self.num_classes = len(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(
            str(self.data_dir / mode),
            class_to_idx,
            ("mp4"),
            is_valid_file=None
        )
        if self.temporal_views == -1 and self.spatial_views == -1:
            self.clips_per_video = list(product([0], [0]))
        elif self.temporal_views == -1:
            self.clips_per_video = list(product([0], range(self.spatial_views)))
        elif self.spatial_views == -1:
            self.clips_per_video = list(product(range(self.temporal_views), [0]))
        else:
            self.clips_per_video = list(product(range(self.temporal_views), range(self.spatial_views)))
        # create a list of tuples (video_path, class_label, temporal_sample_index, spatial_sample_index)
        self.video_clips = []
        for video_idx, (video_path, class_label) in enumerate(self.samples):
            for (temporal_sample_index, spatial_sample_index) in self.clips_per_video:
                self.video_clips.append((video_path, class_label, video_idx, temporal_sample_index, spatial_sample_index))
        print(
            f"Constructing kinetics dataloader ",
            f"(size: {len(self.video_clips)}) from {self.data_dir / mode}"
        ) 


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        success = False
        while not success:
            video_path, _, video_idx, temporal_sample_index, _ = self.video_clips[index]
            try:
                vr = de.VideoReader(str(video_path), num_threads=1, ctx=de.cpu(0))
                num_frames = len(vr)
                # should be random for training but we skip this for now

                if num_frames <= self.num_frames * self.sampling_rate:
                    all_indices = np.linspace(0, num_frames - 1, self.num_frames * self.sampling_rate, dtype=int)
                    all_indices = all_indices[:self.num_frames]
                else:
                    all_indices = np.arange(0, num_frames, self.sampling_rate)
                
                num_posible_views = len(all_indices) // (self.num_frames * self.sampling_rate) + 1 # at leat on view is always possible
                if num_posible_views < self.temporal_views:
                    idx = temporal_sample_index % num_posible_views
                
                reshaped_indices = as_strided(all_indices, shape=(num_posible_views, self.num_frames), strides= all_indices.strides*2)
                indices = reshaped_indices[idx]
                # print(indices)
                vr.seek(0)
                video = from_dlpack(to_dlpack(vr.get_batch(indices)))
                info = {}
                # del vr
                success = True
            except Exception as e:
                print('skipped idx', index, e)
                index = np.random.randint(len(self.video_clips))
        label = self.video_clips[index][1]
        if self.transform is not None:
            video = self.transform(video)
        video_info = {
            'video_label': label,
            'video_index': video_idx,
            'clip_index': temporal_sample_index,
            'info': info
        }
        return video, label, video_info

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.video_clips)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.samples)


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])


def collate_fn(batch):
    # remove audio from the batch keep only video and label
    batch = [(video, label) for video, label, _ in batch]
    return default_collate(batch)


if __name__ == '__main__':
    data_path = Path('/scratch/jeh16/datasets/k400/')
    mode = 'val'
    epochs = 10 # simulate 10 epochs of training

    IMG_MEAN = (0.4914, 0.4822, 0.4465)
    IMG_STD  = (0.2023, 0.1994, 0.2010)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224,  scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    train_transform = MapTransform(train_transform)

    dataset = Kinetics(
        mode=mode,
        data_dir=data_path,
        sampling_rate=4,
        num_frames=16,
        temporal_views=7,
        spatial_views=-1,
        transform=train_transform,
    )

    num_samples = len(dataset)
    print(f"There are {num_samples} in the dataset")

    train_sampler = torch.utils.data.sampler.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=40//2,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    times = []
    for epoch in range(epochs):
        start = time.time()
        for i, (video, target) in enumerate(tqdm(data_loader)):
            pass
        end = time.time()
        times.append(end - start)
        print('epoch', epoch, 'time', end - start)
        # Fisrt epoch ~ 30 min
    print(times)
    print('mean time', np.mean(times), 'std time', np.std(times))
    print('samples per second', num_samples / np.mean(times))