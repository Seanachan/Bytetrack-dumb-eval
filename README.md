# Referring Multi-Object Tracking

This repository uses implementation of the paper [Referring Multi-Object Tracking](https://arxiv.org/abs/2303.03366). More project details can be found in the [website](https://referringmot.github.io/).


## Introduction


**Abstract.** 
Existing referring understanding tasks tend to involve the detection of a single text-referred object. In this paper, we propose a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a  language expression as a semantic cue to guide the prediction of multi-object tracking. To the best of our knowledge, it is the first work to achieve an arbitrary number of referent object predictions in videos. To push forward RMOT, we construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI. Specifically, it provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects. Further, we develop a transformer-based architecture TransRMOT to tackle the new task in an online manner, which achieves impressive detection performance.


## Getting started
### Installation


#### Requirements

- **Python**: 3.8
- **PyTorch**: 2.0.1 (with CUDA 11.8 support)
- **torchvision**: 0.15.2+cu118
- **CUDA Toolkit**: 11.3+
- **NumPy**: 1.23.5
- **Transformers**: 4.30.0 (for RoBERTa)

#### Installation Steps

1. **Create a conda environment:**
   ```shell
   conda create -n rmot python=3.8
   conda activate rmot
   ```

2. **Install PyTorch with CUDA support:**
   ```shell
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install MultiScaleDeformableAttention:**
   ```shell
   cd models/ops
   python setup.py build install
   cd ../..
   ```

4. **Install all dependencies from requirements.txt:**
   ```shell
   pip install -r requirements.txt
   ```

The `requirements.txt` includes:
- **Core tracking/detection**: pycocotools, motmetrics, lap
- **Data processing**: tqdm, cython, scipy, opencv-python, seaborn
- **ByteTrack integration**: requests, loguru
- **LLM integration**: pillow, einops, transformers==4.30.0

### Dataset
You can download [expression created by original creator](https://github.com/wudongming97/RMOT/releases/download/v1.0/expression.zip) and [labels_with_ids](https://github.com/wudongming97/RMOT/releases/download/v1.0/labels_with_ids.zip). 
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
- Expression (.json) contains corresponding object ids, and the corresponding boxes can be found in 'labels_with_ids' using these ids.
- The 'label_with_ids' is generated from a script from folder `tools`.
But we strongly recommend **not** using it because the generated track_id may not correspond the track_id of our expression files.

### Modifying Data Paths


#### For Testing
To use custom test data, modify the following parameter in `configs/r50_rmot_test.sh`:

1. **RMOT base path** (Line 28):
   ```shell
   --rmot_path /home/seanachan/RMOT
   ```
   Update this to match your RMOT installation directory where the test data is located.

#### For Evaluation
To evaluate tracking results, modify the following parameters in `TrackEval/scripts/evaluate_rmot.sh`:

1. **Sequence map file** (Line 16):
   ```shell
   --SEQMAP_FILE /home/seanachan/RMOT/datasets/data_path/seqmap_existing.txt
   ```
   This file lists all sequences to evaluate (format: `video_id+expression_name` per line).

2. **Ground truth folder** (Line 18):
   ```shell
   --GT_FOLDER /home/seanachan/RMOT/exps/default/results_epoch99
   ```
   Base path containing ground truth annotations in sub-directories.

3. **Tracker results folder** (Line 19):
   ```bash
   --TRACKERS_FOLDER /home/seanachan/RMOT/exps/bytetrack_llm
   ```
   Path to directory containing tracking predictions from ByteTrack or other trackers.

4. **Ground truth location format** (Line 20):
   ```bash
   --GT_LOC_FORMAT {gt_folder}/{video_id}/{expression_id}/gt.txt
   ```
   This defines the directory structure for ground truth files.

**Note:** Ensure frame numbers in both `gt.txt` and `predict.txt` files are sequential (1 to N) and match the expected sequence length. Frame number mismatches will cause evaluation errors.


### Additional Files (Not in Repository)

Some large files are not tracked in this repository to keep it lightweight. These include:

- **Pre-trained checkpoint**: `exps/default/checkpoint0099.pth` (968MB)
- **Experiment results**: Files in `exps/default/results_epoch99/` and `exps/llm_filtered/`

**Download and Setup:**

1. Download `untracked_files.zip` from [Google Drive](https://drive.google.com/file/d/1T7qsNxkvf9VNkaPKPjXxqbOHKFT872t9/view?usp=drive_link)
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

## ByteTrack with LLM Filtering

This section explains the ByteTrack integration with LLM-based target filtering for RMOT.

### Overview

The system uses ByteTrack (from `~/ByteTrack_LLM_dumb`) as the object detector and tracker, combined with an LLM to identify which tracked object matches a natural language description.

### Key Components

#### 1. ByteTrack Bounding Box Storage

Bounding boxes from ByteTrack are stored in multiple locations:

- **In-Memory**: `Instances.boxes` - PyTorch tensor in xyxy format `[x1, y1, x2, y2]`
- **Track Objects**: `Track.box` - Current bounding box for each tracked object
- **Output Files**: `predict.txt` - MOT Challenge format `frame_id, track_id, x1, y1, width, height, ...`

#### 2. LLM Filter Integration Flow

The inference pipeline works as follows:

```
Frame 0 (First Frame):
  1. Run ByteTrack detection → get multiple detections
  2. For each detection:
     - Crop the detection region from the image
     - Send cropped image + natural language description to LLM
     - LLM responds: "yes" (matches) or "no" (doesn't match)
  3. Store the ID of the first matching object (target_id)

Frame 1+ (Subsequent Frames):
  1. Run ByteTrack detection → get multiple detections
  2. Filter detections: keep only those with ID == target_id
  3. Write filtered results to output file
```

#### 3. LLM Prompt and Image Cropping

The LLM filter uses cropped sub-images instead of full frames:

- **Original Image**: Full video frame (e.g., 1920×1080)
- **Detection**: ByteTrack returns bounding box `[100, 200, 150, 280]` (coordinates in full frame)
- **Crop**: Extract region from image: `image[y1:y2, x1:x2]` → smaller image (e.g., 50×80)
- **LLM Input**: Cropped image + prompt like "Is this a red car?"
- **LLM Output**: "yes" or "no"

This approach is more efficient and focused than sending full frames to the LLM.

#### 4. First Frame Detection Logic

The system automatically detects the first frame using a loop counter:

```python
for frame_id, (cur_img, ori_img) in enumerate(tqdm(loader)):
    if frame_id == 0 and len(dt_instances) > 0:
        # First frame: run LLM filtering
        target_id = llm_filter.filter_first_frame(...)
```

`frame_id == 0` on the first iteration, triggering LLM filtering only once.

#### 5. Inference with ByteTrack

Run inference with ByteTrack and LLM filtering:

```shell
python inference_bytetrack_llm.py \
  --rmot_path /home/seanachan/RMOT \
  --bytetrack_exp /home/seanachan/ByteTrack_LLM_dumb/exps/example/mot/yolox_x_mix_det.py \
  --bytetrack_ckpt /home/seanachan/ByteTrack_LLM_dumb/pretrained/bytetrack_x_mot17.pth.tar \
  --output_dir exps/bytetrack_llm \
  --video_id 0013 \
  --expression_json 0.json \
  --llm_api_url http://localhost:11434/api/generate \
  --llm_model qwen2.5vl
```

Or use the config script:
```shell
sh configs/bytetrack_llm_test.sh
```

You should be seeing:
```shell
============================================================
[INFO] Evaluating seq ['0005', 'black-vehicles-in-the-left.json']
============================================================
[INFO] Results will be saved to exps/bytetrack_llm/0005/black-vehicles-in-the-left
Processing 297 frames with sentence: "black vehicles in the left"
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 297/297 [00:44<00:00,  6.64it/s]
[INFO] Total 39 detections written to exps/bytetrack_llm/0005/black-vehicles-in-the-left/predict.txt
```

**Output Structure:**
```
exps/bytetrack_llm/
  └── {video_id}/
      └── {expression_id}/
          ├── predict.txt        ← Tracking predictions
          └── crops/             ← LLM debug crops (first frame detections)
              ├── i0_f0.jpg
              ├── i1_f0.jpg
              └── ...
```

### Evaluation

#### Understanding Evaluation Parameters

For evaluation, TrackEval uses these key parameters:

- **`--GT_FOLDER`**: Base directory containing ground truth files
  - Example: `/home/seanachan/RMOT/KITTI/training`
  
- **`--TRACKERS_FOLDER`**: Directory containing your tracking predictions
  - Example: `/home/seanachan/RMOT/exps/bytetrack_llm`
  
- **`--GT_LOC_FORMAT`**: Pattern to construct full path to GT files
  - Example: `{gt_folder}/labels_with_ids/image_02/{video_id}/{expression_id}/gt.txt`
  - `{gt_folder}` replaced with `--GT_FOLDER`
  - `{video_id}` and `{expression_id}` auto-discovered from folder structure
  - Full path: `/home/seanachan/RMOT/KITTI/training/labels_with_ids/image_02/0013/0/gt.txt`
  
- **`--TRACKERS_TO_EVAL`**: Which tracker subdirectory to evaluate
  - Example: `bytetrack_llm` (evaluates `/TRACKERS_FOLDER/bytetrack_llm`)

#### Ground Truth vs Predictions

- **`gt.txt`**: Ground truth file (the correct answer)
  - Contains actual object positions and IDs that should be tracked
  - Format: `frame_id, track_id, x1, y1, width, height, ...`
  
- **`predict.txt`**: Your method's output (what your algorithm predicted)
  - Contains detections/tracks from ByteTrack+LLM
  - Same format as gt.txt

TrackEval compares these files and calculates metrics:
- **HOTA**: How well you tracked objects over time
- **CLEAR**: Detection accuracy (precision, recall)
- **Identity**: How well you maintained consistent track IDs

#### Running Evaluation

Update paths in `TrackEval/scripts/evaluate_rmot.sh`:

```shell
python3 run_mot_challenge.py \
--METRICS HOTA \
--GT_FOLDER /home/seanachan/RMOT/KITTI/training \
--TRACKERS_FOLDER /home/seanachan/RMOT/exps/bytetrack_llm \
--GT_LOC_FORMAT {gt_folder}/labels_with_ids/image_02/{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL bytetrack_llm \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False
```

Then run evaluation:
```shell
cd TrackEval/scripts
bash evaluate_rmot.sh
```

## Results (under construction)


The main results of TransRMOT:

| **Method** | **Dataset** | **HOTA** | **DetA** | **AssA** | **DetRe** | **DetPr** | **AssRe** | **AssRe** | **LocA** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| TransRMOT  | Refer-KITTI |  38.06   |  29.28   |  50.83   |   40.19   |   47.36   |   55.43   | 81.36     | 79.93    | [model](https://github.com/wudongming97/RMOT/releases/download/v1.0/checkpoint0099.pth) |


We also provide [FairMOT results](https://github.com/wudongming97/RMOT/releases/download/v1.0/FairMOT_results.zip) as references.



## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)


