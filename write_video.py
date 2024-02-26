import os
import numpy as np

import torch
from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBVideoField # , JSONField

import torchvision.datasets.video_utils
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2

# # Write Kinetics-400 dataset
# python write_video.py --cfg.dataset k400 --cfg.split training --cfg.data_dir /scratch/jeh16/datasets/k400/ --cfg.write_path /scratch/jeh16/ffcv/k400/k400_train_16x4.ffcv --cfg.min_resolution 320 --cfg.quality 90 --cfg.max_frames 16 --cfg.frame_skip 4 --cfg.num_workers 50

# python write_video.py --cfg.dataset k400 --cfg.split validation --cfg.data_dir /scratch/jeh16/datasets/k400/ --cfg.write_path /scratch/jeh16/ffcv/k400/k400_val_16x4_1x5.ffcv --cfg.min_resolution 320 --cfg.quality 90 --cfg.max_frames 16 --cfg.frame_skip 4 --cfg.num_workers 50


class VideoDataset(VisionDataset):
    """
    Generic Video dataset where the video is sampled at a fixed frame rate.
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
        super(VideoDataset, self).__init__(root)
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

        # video to from TCHW to THWC
        video = video.permute(0, 2, 3, 1)
        video = video.numpy()

        return video, label, video_idx, clip_idx


def _get_cache_path(filepath, dataset):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", dataset, h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['mit', 'k400', 'k600', 'k700'])), 'Which dataset to write', default='mit'),
    split=Param(And(str, OneOf(['training', 'validation', 'testing'])), 'Train or val set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    min_resolution=Param(int, 'Min image side length', required=True),
    frame_skip=Param(int, 'How many frames to skip', default=4),
    max_frames=Param(int, 'Max number of frames', required=True),
    num_workers=Param(int, 'Number of workers to use', default=40),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
)


@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('write_path')
@param('min_resolution')
@param('frame_skip')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('quality')
@param('max_frames')
def main(
    dataset,
    split,
    data_dir,
    write_path,
    min_resolution,
    frame_skip,
    num_workers,
    chunk_size,
    subset,
    quality,
    max_frames,
):

    data_dir = Path(data_dir)
    
    if dataset == 'mit':
        mode_map = {
            'training': 'training',
            'validation': 'validation',
            'testing': 'testing'
        }

    elif dataset in ['k400', 'k600', 'k700']:
        mode_map = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }
    else:
        raise ValueError('Unrecognized dataset', dataset)    

    cache_path = _get_cache_path(str(data_dir / mode_map[split]), dataset)
    # this will sample each clip from the video using a sliding window approach with stride 2
    # when validating or testing, we want to see a disjoint set of clips so we use max_frames
    #  as the step_between_clips so that we get all the disjoint clips
    step_between_clips = max_frames // 2 if split == 'training' else max_frames

    if os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        cached = dict(
            video_paths=dataset.video_clips.video_paths,
            video_fps=dataset.video_clips.video_fps,
            video_pts=dataset.video_clips.video_pts
        )
        dataset = VideoDataset(
            root=str(data_dir / mode_map[split]),
            frames_per_clip=max_frames,
            step_between_clips=step_between_clips,
            transform=None,
            num_workers=num_workers,
            _precomputed_metadata=cached,
        )
    else:
        dataset = VideoDataset(
            root=str(data_dir / mode_map[split]),
            frames_per_clip=max_frames,
            step_between_clips=step_between_clips,
            transform=None,
            num_workers=num_workers,
            _precomputed_metadata=None,
        )
        print("Saving dataset_train to {}".format(cache_path))
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        torch.save((dataset, str(data_dir / mode_map[split])), cache_path)

    # frame_rates = [fps for fps in dataset.video_clips.video_fps if fps is not None]
    # avg_frame_rate = np.mean(frame_rates)
    avg_frame_rate = 30
    # we want to change the frame rate to that the model skip frame_skip frames
    new_frame_rate = np.ceil(avg_frame_rate / frame_skip)
    print(f"Average frame rate: {avg_frame_rate}, new frame rate: {new_frame_rate}")

    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(max_frames, step_between_clips, frame_rate=new_frame_rate)

    print(f"There are {len(dataset)} in the dataset")

    if subset > 0: dataset = Subset(dataset, range(subset))

    writer = DatasetWriter(write_path, {
        'video': RGBVideoField(
            min_resolution=min_resolution,
            quality=quality,
        ),
        'label': IntField(),
        'video_index': IntField(),
        'clip_index': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(dataset, chunksize=chunk_size)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
