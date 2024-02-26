import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

import torch.utils.data
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import time
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler
from PIL import Image

import numpy as np
import os
from typing import Any, Callable, Dict, Optional, Tuple
from tqdm import tqdm


class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """
    def __init__(
            self,
            root,
            frames_per_clip: int,
            frame_rate: Optional[int] = None,
            step_between_clips: int = 1,
            transform: Optional[Callable] = None,
            extensions: Tuple[str, ...] = ("avi", "mp4"),
            num_workers: int = 1,
            _precomputed_metadata: Optional[Dict[str, Any]] = None,
            _video_width: int = 0,
            _video_height: int = 0,
            _video_min_dimension: int = 0,
            _audio_samples: int = 0,
            _audio_channels: int = 0,
            output_format: str = "TCHW",
        ):
        super(Kinetics400, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        self.num_classes = len(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
            output_format=output_format,
        )
        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video_idx, clip_idx = self.video_clips.get_clip_location(idx)
                video, _, info, video_idx = self.video_clips.get_clip(idx)
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)

        video_info = {
            'video_label': label,
            'video_index': video_idx,
            'clip_index': clip_idx,
            'info': info
        }

        return video, label, video_info


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            # vid = vid.permute(0, 2, 3, 1)
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch keep only video and label
    batch = [(video, label) for video, label, _ in batch]
    return default_collate(batch)


if __name__ == '__main__':
    data_path = Path('/scratch/jeh16/datasets/k400/')
    traindir = data_path / 'train' # for testing
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

    cache_path = _get_cache_path(str(traindir))

    if os.path.exists(cache_path):
        start = time.time()
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        cached = dict(
            video_paths=dataset.video_clips.video_paths,
            video_fps=dataset.video_clips.video_fps,
            video_pts=dataset.video_clips.video_pts
        )
        dataset = Kinetics400(
            root=str(traindir),
            frames_per_clip=16,
            step_between_clips=1,
            transform=train_transform,
            num_workers=20,
            _precomputed_metadata=cached,
        )
        dataset.transform = train_transform
        end = time.time()
        print("Time taken to load dataset_train: ", end - start)
    else:
        start = time.time()
        dataset = Kinetics400(
            root=str(traindir),
            frames_per_clip=16,
            step_between_clips=1,
            transform=train_transform,
            num_workers=20,
            _precomputed_metadata=None,
        )
        print("Saving dataset_train to {}".format(cache_path))
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        dataset.transform = None
        torch.save((dataset, traindir), cache_path)
        dataset.transform = train_transform
        end = time.time()
        print("Time taken to save dataset_train: ", end - start)

    start = time.time()
    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(16, 2, frame_rate=8)
    end = time.time()
    print("Time taken to compute clips: ", end - start)  

    num_samples = len(dataset)
    print(f"There are {num_samples} in the dataset")

    if hasattr(dataset, 'video_clips'):
        train_sampler = RandomClipSampler(dataset.video_clips, 5)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=40//2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    times = []
    for epoch in range(epochs):
        start = time.time()
        for i, (video, target) in enumerate(tqdm(data_loader)):
            # Code to run the model
            pass
        end = time.time()
        times.append(end - start)
        print('epoch', epoch, 'time', end - start)

    print(times)
    print('mean time', np.mean(times), 'std time', np.std(times))
    print('samples per second', num_samples / np.mean(times))