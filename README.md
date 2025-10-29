# Referring Multi-Object Tracking

This repository is an official implementation of the paper [Referring Multi-Object Tracking](https://arxiv.org/abs/2303.03366). More project details can be found in the [website](https://referringmot.github.io/).

### :heavy_exclamation_mark: :heavy_exclamation_mark: :heavy_exclamation_mark: The bootstrapped version of this repo is released. Please see [TempRMOT](https://github.com/zyn213/TempRMOT).

## Introduction


<div style="align: center">
<img src=./figs/TransRMOT.png/>
</div>

**Abstract.** 
Existing referring understanding tasks tend to involve the detection of a single text-referred object. In this paper, we propose a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a  language expression as a semantic cue to guide the prediction of multi-object tracking. To the best of our knowledge, it is the first work to achieve an arbitrary number of referent object predictions in videos. To push forward RMOT, we construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI. Specifically, it provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects. Further, we develop a transformer-based architecture TransRMOT to tackle the new task in an online manner, which achieves impressive detection performance.

## Updates
- (2023/03/19) RMOT dataset and code are released.
- (2023/03/07) RMOT paper is available on [arxiv](https://arxiv.org/abs/2303.03366).
- (2023/02/28) RMOT is accepted by CVPR2023! The dataset and code is coming soon!



## Getting started
### Installation

The basic environment setup is on top of [MOTR](https://github.com/megvii-research/MOTR), including conda environment, pytorch version and other requirements.

#### Requirements

- **Python**: 3.8
- **PyTorch**: 2.0.1 (with CUDA 11.8 support)
- **torchvision**: 0.15.2+cu118
- **CUDA Toolkit**: 11.3+
- **NumPy**: 1.23.5
- **Transformers**: 4.30.0 (for RoBERTa)

#### Installation Steps

1. Create a conda environment:
   ```bash
   conda create -n rmot python=3.8
   conda activate rmot
   ```

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install MultiScaleDeformableAttention:
   ```bash
   cd models/ops
   python setup.py build install
   cd ../..
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   pip install transformers==4.30.0
   pip install einops pillow
   ```

The `requirements.txt` includes:
- pycocotools
- tqdm
- cython
- scipy
- motmetrics
- opencv-python
- seaborn
- lap

### Dataset
You can download [our created expression](https://github.com/wudongming97/RMOT/releases/download/v1.0/expression.zip) and [labels_with_ids](https://github.com/wudongming97/RMOT/releases/download/v1.0/labels_with_ids.zip). 
The KITTI images are from [official website](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), which are unzipped into `./KITTI/training`.
The Refer-KITTI is organized as follows:

```
.
├── refer-kitti
│   ├── KITTI
│           ├── training
│           ├── labels_with_ids
│   └── expression
```
Note: 
- Our expression (.json) contains corresponding object ids, and the corresponding boxes can be found in 'labels_with_ids' using these ids.
- The 'label_with_ids' is generated from a script from folder `tools`.
But we strongly recommend **not** using it because the generated track_id may not correspond the track_id of our expression files.

### Modifying Data Paths

#### For Training
To use custom training data, modify the following parameters in `configs/r50_rmot_train.sh`:

1. **Data list file** (Line 35):
   ```bash
   --data_txt_path_train ./datasets/data_path/refer-kitti.train
   ```
   Change this to point to your custom data list file. The file should contain relative paths to images (one per line).
   Format: `KITTI/training/image_02/[sequence]/[frame].png`

2. **RMOT base path** (Line 34):
   ```bash
   --rmot_path /home/seanachan/RMOT
   ```
   Update this to your RMOT installation directory where the dataset is located.

#### For Testing
To use custom test data, modify the following parameter in `configs/r50_rmot_test.sh`:

1. **RMOT base path** (Line 28):
   ```bash
   --rmot_path /home/seanachan/RMOT
   ```
   Update this to match your RMOT installation directory where the test data is located.

#### For Evaluation
To evaluate tracking results, modify the following parameters in `TrackEval/scripts/evaluate_rmot.sh`:

1. **Sequence map file** (Line 16):
   ```bash
   --SEQMAP_FILE /home/seanachan/RMOT/datasets/data_path/seqmap_existing.txt
   ```
   This file lists all sequences to evaluate (format: `video_id+expression_name` per line).

2. **Ground truth folder** (Line 18):
   ```bash
   --GT_FOLDER /home/seanachan/RMOT/exps/default/results_epoch99
   ```
   Path to directory containing ground truth annotations.

3. **Tracker results folder** (Line 19):
   ```bash
   --TRACKERS_FOLDER /home/seanachan/RMOT/exps/default/results_epoch99
   ```
   Path to directory containing tracking predictions.

4. **Ground truth location format** (Line 20):
   ```bash
   --GT_LOC_FORMAT {gt_folder}/{video_id}/{expression_id}/gt.txt
   ```
   This defines the directory structure for ground truth files.

**Note:** Ensure frame numbers in both `gt.txt` and `predict.txt` files are sequential (1 to N) and match the expected sequence length. Frame number mismatches will cause evaluation errors.

### Training
You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) ''+ iterative bounding box refinement''.
Then training TransRMOT on 8 GPUs as following:
```bash 
sh configs/r50_rmot_train.sh
```
Note:
- If the RoBERTa is not working well, please download the RoBERTa weights from [Hugging Face](https://huggingface.co/roberta-base/tree/main) for local using.

### Testing
You can download the pretrained model of TransRMOT (the link is in "Main Results" session), then run following command to generate and save prediction boxes:
```bash
sh configs/r50_rmot_test.sh
```

You can get the main results by runing the evaluation part. You can also use our [prediction and gt file](https://github.com/wudongming97/RMOT/releases/download/v1.0/results_epoch99.zip). 
```bash
cd TrackEval/script
sh evaluate_rmot.sh
```

### Additional Files (Not in Repository)

Some large files are not tracked in this repository to keep it lightweight. These include:

- **Pre-trained checkpoint**: `exps/default/checkpoint0099.pth` (968MB)
- **Experiment results**: Files in `exps/default/results_epoch99/` and `exps/llm_filtered/`

**Download and Setup:**

1. Download `untracked_files.zip` from [Google Drive](INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE)
2. Extract the zip file:
   ```bash
   unzip untracked_files.zip
   ```
3. Copy only the `exps/` directory to your project root:
   ```bash
   cp -r RMOT_untracked_files/exps /path/to/your/RMOT/
   ```
   Or if you're already in the project directory:
   ```bash
   cp -r RMOT_untracked_files/exps .
   ```

This will restore the checkpoint file and experiment results without affecting other files in your repository.

## Results


The main results of TransRMOT:

| **Method** | **Dataset** | **HOTA** | **DetA** | **AssA** | **DetRe** | **DetPr** | **AssRe** | **AssRe** | **LocA** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| TransRMOT  | Refer-KITTI |  38.06   |  29.28   |  50.83   |   40.19   |   47.36   |   55.43   | 81.36     | 79.93    | [model](https://github.com/wudongming97/RMOT/releases/download/v1.0/checkpoint0099.pth) |


We also provide [FairMOT results](https://github.com/wudongming97/RMOT/releases/download/v1.0/FairMOT_results.zip) as references.



## Citing RMOT
If you find RMOT useful in your research, please consider citing:

```bibtex
@inproceedings{wu2023referring,
  title={Referring Multi-Object Tracking},
  author={Wu, Dongming and Han, Wencheng and Wang, Tiancai and Dong, Xingping and Zhang, Xiangyu and Shen, Jianbing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14633--14642},
  year={2023}
}
```


## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)


