# Video datasets for PyTorch.
This code organizes the a standard video classification dataset into a format that is easy to use with PyTorch. We try several methods to load the Kinetics-400 dataset (241255 videos in train, 19881 videos in validations), and compare them in terms of speed and resulting size. These methods are:
- The [standard method](https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py), which is to store the videos using PyTorch's VideoReader class. We use this method as a baseline. This just stores the videos as they are, mp4 files.
- Using [torchvision.IO](https://github.com/facebookresearch/mae_st/blob/main/util/kinetics.py) to load the videos. This is a library that uses the torchvision's VideoReader class and PyAV to load the videos (I was unable to make the new API work see [here](https://github.com/pytorch/vision/issues/5720)). This just requires the videos to be stored as mp4 files.
- Using [decord](https://github.com/dmlc/decord) to load the videos. This is a library that is supposed to be faster than PyTorch's VideoReader. This just requires the videos to be stored as mp4 files.
- Using the [python bindings of ffmpeg](https://github.com/kkroening/ffmpeg-python) to load the videos. This just requires the videos to be stored as mp4 files.
- Using [FFCV](https://github.com/libffcv/ffcv), we store the videos as a sequence of JPEG images. This is requires several preprocessing steps, (1) We extract the frames from the videos, (2) We resize the frames to a maximum size, (3) We encode the frames as JPEG images, (4) We store the frames in a single FFCV file.

Since videos are most of the time redundat, people usually skip frames when giving them to a neural network. This is ussally denoted as Txt, where T is the number of frames the model sees and t is the number of frames that are skipped in between. We use T=16 and t=4, which is the standard for Kinetics-400. Similarly, for evaluation people usually try to take full coverage of the video, so they split the video into several crops (this can be overlapin or non-overlaping) and average the predictions. Crops are denoted as sxτ, where s is the number of spatial crops and τ is the number of time crops. We use 5 time crops an 1 spatial crop, there is no standard and this varies from paper to paper.

## Requirements

- Download the Kinetics dataset from https://github.com/cvdfoundation/kinetics-dataset
- Move and extract the training and validation videos to labeled subfolders, using [the following shell script](reorganize_k400.py)
- Create the FFCV file using the following code:
```bash
python write_video.py --cfg.dataset k400 --cfg.split training --cfg.data_dir /scratch/jeh16/datasets/k400/ --cfg.write_path /scratch/jeh16/ffcv/k400/k400_train_16x4.ffcv --cfg.min_resolution 320 --cfg.quality 90 --cfg.max_frames 16 --cfg.frame_skip 4 --cfg.num_workers 50
```
```bash
python write_video.py --cfg.dataset k400 --cfg.split validation --cfg.data_dir /scratch/jeh16/datasets/k400/ --cfg.write_path /scratch/jeh16/ffcv/k400/k400_val_16x4_1x5.ffcv --cfg.min_resolution 320 --cfg.quality 90 --cfg.max_frames 16 --cfg.frame_skip 4 --cfg.num_workers 50
```

## Video loading
We simulate the video loading process by loading the videos and skipping frames. We always perform 10 measurements and report the average. We perform the following standard augmentations RamdomResizedCrop->RandomHorizontalFlip->Normalize. We use the following parameters: T=16, t=4, s=1, τ=5. We use a batch size of 32 ans set the numer of workers to 40. We use a single NVIDIA A40 GPU.
- We use the following code to load the videos using the standard method:
```bash
python kinetics_legacy.py 
```
- We use the following code to load the videos using torchvision.IO:
```bash
python kinetics_pyav.py
```
- We use the following code to load the videos using decord:
```bash
python kinetics_decord.py
```
- We use the following code to load the videos using ffmpeg:
```bash
python kinetics_ffmpeg.py
```
- We use the following code to load the videos using FFCV:
```bash
python kinetics_ffcv.py
```

## Speed Results

| Library                | Train Size (GB) | Val Size (GB) | Num_train | Num_val | Frames | Skip | videos/Second |
|------------------------|-----------------|---------------|-----------|---------|--------|------|---------------|
| VideClips (Torchvision)| 348             | 29            | 241255    | 19881   | 16     | 4    | 276.55        |
| Torchvision.IO (PyAV)              | 348             | 29            | 241255    | 19881   | 16     | 4    | 42.36         |
| Decord                 | 348             | 29            | 241255    | 19881   | 16     | 4    | 70.75         |
| FFmpeg                 | 348             | 29            | 241255    | 19881   | 16     | 4    | 26.91         |
| FFCV                   | 946             | 44            | 241255    | 19881   | 16     | 4    | 564.03        |

## Model evaluations using FFCV:
The models on the HUB that I found are:
- [VideoMAE](https://arxiv.org/abs/2203.12602) on the following configurations: [Small](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics), [Base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics), [Large](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics), [Huge](https://huggingface.co/MCG-NJU/videomae-huge-finetuned-kinetics). Trained using 16 frames and skipping 4.
- My own [ViC-MAE](https://arxiv.org/abs/2303.12001) on the following configurations: Base, Large. See the ViC-MAE [repository](https://github.com/jeffhernandez1995/ViC-MAE) for more details. Trained using 16 frames and skipping 4.
- [TimeSformer](https://arxiv.org/pdf/2102.05095.pdf) on the following configurations: [Base](https://huggingface.co/facebook/timesformer-base-finetuned-k400), [HR](https://huggingface.co/facebook/timesformer-hr-finetuned-k400). Trained using 8 frames and skipping 4 and 16 frames and skipping 4, respectively.
- [ViViT](https://arxiv.org/pdf/2103.15691.pdf) on the following configurations: [Base](https://huggingface.co/google/vivit-b-16x2-kinetics400). Trained using 32 frames and skipping 2.


The results we obtain are:
| Name         | Arch    | Fxs  | SxT views | Accuracy | SxT views | Accuracy |
|--------------|---------|------|-----------|----------|-----------|----------|
| VideoMAE     | ViT/S-16| 16x4 | 3x5       | 79       | 1x5       | 54.7     |
| VideoMAE     | ViT/B-16| 16x4 | 3x5       | 81.5     | 1x5       | 77.7     |
| VideoMAE     | ViT/L-14| 16x4 | 3x5       | 85.2     | 1x5       | 81.9     |
| VideoMAE     | ViT/H-14| 16x4 | 3x5       | 86.6     | 1x5       | 83       |
| ViCMAE       | ViT/B-16| 16x4 | 3x7       | 80.8     | 1x5       | 77.3     |
| ViCMAE       | ViT/L-14| 16x4 | 3x7       | 86.8     | 1x5       | 83.1     |
| TimesFormer  | Base    | 8x4  | 3x1       | 79.1     | 1x5       | 73.9     |
| TimesFormer  | HR      | 16x4 | 3x1       | 81.8     | 1x5       | 64.3     |
| ViViT        | ViViT/B | 32x2 | 1x4       | 79.9     | 1x5       | 66.5     |
