"""
ByteTrack detector wrapper for RMOT inference with LLM filtering
Wraps ByteTrack from ~/ByteTrack_LLM_dumb and converts detections to Instances format
"""
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# Add ByteTrack to path
BYTETRACK_ROOT = Path.home() / "ByteTrack_LLM_dumb"
sys.path.insert(0, str(BYTETRACK_ROOT))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracker.byte_tracker import BYTETracker

from models.structures import Instances


class ByteTrackDetector:
    """Wrapper around ByteTrack that returns detections in Instances format"""
    
    def __init__(self, exp_file, ckpt_file, device="gpu", fp16=False, fuse=True, 
                 track_thresh=0.5, track_buffer=30, match_thresh=0.8, fps=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fps = fps
        """
        Args:
            exp_file: Path to experiment config file
            ckpt_file: Path to model checkpoint
            device: "gpu" or "cpu"
            fp16: Use half precision
            fuse: Fuse conv+bn
            track_thresh: Tracking confidence threshold
            track_buffer: Frames to keep lost tracks
            match_thresh: Matching threshold
            fps: Video frame rate
        """
        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()
        
        # Load checkpoint
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        
        # Setup device
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Fuse and eval
        if fuse:
            self.model = fuse_model(self.model)
        self.model.eval()
        
        # Store params
        self.fp16 = fp16
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # ByteTrack tracker
        class BytetrackArgs:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh

        args = BytetrackArgs()
        self.tracker = BYTETracker(args, frame_rate=fps)
        
        self.frame_id = 0
    
    def detect(self, ori_img):
        """
        Detect objects in a single frame
        
        Args:
            ori_img: Original image (numpy array, H x W x 3, RGB format)
            
        Returns:
            Instances object with fields: boxes, scores, obj_idxes (track IDs)
        """
        height, width = ori_img.shape[:2]
        
        # Preprocess
        img, ratio = preproc(ori_img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        
        # Free input tensor memory immediately
        del img
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Update tracker
        self.frame_id += 1
        online_targets = self.tracker.update(outputs[0], [height, width], self.test_size)
        
        # Convert to Instances format
        boxes = []
        scores = []
        track_ids = []
        
        for t in online_targets:
            tlwh = t.tlwh  # [x, y, w, h]
            # Convert to xyxy format
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            boxes.append([x1, y1, x2, y2])
            scores.append(t.score)
            track_ids.append(t.track_id)
        
        # Create Instances object
        instances = Instances(image_size=(height, width))
        
        if len(boxes) > 0:
            instances.boxes = torch.as_tensor(np.array(boxes, dtype=np.float32))
            instances.scores = torch.as_tensor(np.array(scores, dtype=np.float32))
            instances.obj_idxes = torch.as_tensor(np.array(track_ids, dtype=np.int64))
        else:
            instances.boxes = torch.zeros((0, 4), dtype=torch.float32)
            instances.scores = torch.zeros((0,), dtype=torch.float32)
            instances.obj_idxes = torch.zeros((0,), dtype=torch.int64)
        
        return instances
    
    def reset(self):
        """Reset tracker for new video"""
        class BytetrackArgs:
            def __init__(self, track_thresh, track_buffer, match_thresh):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh

        args = BytetrackArgs(self.track_thresh, self.track_buffer, self.match_thresh)
        self.tracker = BYTETracker(args, frame_rate=self.fps)
        self.frame_id = 0
