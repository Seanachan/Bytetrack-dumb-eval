"""
RMOT Inference with ByteTrack and LLM Filtering
Uses ByteTrack detector instead of TransRMOT for faster inference
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root and ByteTrack to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(str(Path.home()), 'ByteTrack_LLM_dumb'))

from bytetrack_detector import ByteTrackDetector
from llm_filter import LLMFilter
from inference import ListImgDataset, Detector  # Reuse these utilities
from models.structures import Instances

# Use write_results from Detector
write_results = Detector.write_results


class ByteTrackLLMInference:
    """ByteTrack-based inference with LLM filtering"""
    
    def __init__(self, detector, output_dir, rmot_path, seq_num, 
                 llm_api_url="http://localhost:11434/api/generate", 
                 llm_model="qwen2.5vl"):
        """
        Args:
            detector: Shared ByteTrackDetector instance
            output_dir: Output directory for results
            rmot_path: Root RMOT data path
            seq_num: [video_id, expression_json] tuple
            llm_api_url: LLM API endpoint
            llm_model: LLM model name
        """
        self.output_dir = output_dir
        self.rmot_path = rmot_path
        self.seq_num = seq_num
        
        # Setup paths
        img_list = os.listdir(os.path.join(rmot_path, 'KITTI/training/image_02', seq_num[0]))
        img_list = [os.path.join(rmot_path, 'KITTI/training/image_02', seq_num[0], _)
                   for _ in img_list if ('jpg' in _) or ('png' in _)]
        self.img_list = sorted(img_list)
        
        # Load JSON with natural language expression
        json_path = os.path.join(rmot_path, 'expression', seq_num[0], seq_num[1])
        with open(json_path, 'r') as f:
            json_info = json.load(f)
        self.sentence = json_info['sentence']
        
        # Setup save paths
        self.save_path = os.path.join(output_dir, seq_num[0], seq_num[1].split(".")[0])
        os.makedirs(self.save_path, exist_ok=True)
        
        # Use shared detector
        self.detector = detector
        
        # Initialize LLM filter
        self.llm_filter = LLMFilter(
            api_url=llm_api_url,
            model=llm_model,
            crops_dir=os.path.join(self.save_path, 'crops')
        )
        
        self.target_id = None
        print(f"[INFO] Results will be saved to {self.save_path}")
    
    def detect(self):
        """Run detection and tracking on all frames"""
        print(f'Processing {len(self.img_list)} frames with sentence: "{self.sentence}"')
        
        loader = DataLoader(ListImgDataset(self.img_list), 1, num_workers=0)
        total_detections = 0
        
        # Clear output file
        predict_txt = os.path.join(self.save_path, 'predict.txt')
        if os.path.exists(predict_txt):
            os.remove(predict_txt)
        
        for frame_id, (cur_img, ori_img) in enumerate(tqdm(loader)):
            ori_img = ori_img[0]  # Extract from batch
            seq_h, seq_w, _ = ori_img.shape
            
            # ByteTrack detection
            dt_instances = self.detector.detect(ori_img.numpy())
            
            # LLM filtering on first frame
            if frame_id == 0 and len(dt_instances) > 0:
                print(f"\n[LLM] First frame: {len(dt_instances)} detections before LLM filter")
                
                self.target_id = self.llm_filter.filter_first_frame(
                    image=ori_img.numpy(),
                    boxes=dt_instances.boxes.numpy(),
                    obj_ids=dt_instances.obj_idxes.numpy(),
                    sentence=self.sentence,
                    save_crops=True
                )
                
                if self.target_id is not None:
                    print(f"[LLM] Target locked: ID={self.target_id}\n")
                else:
                    print(f"[LLM] WARNING: No target matched! Tracking all objects.\n")
            
            # Filter to keep only target if LLM found one
            if self.target_id is not None:
                keep_mask = dt_instances.obj_idxes == self.target_id
                dt_instances = dt_instances[keep_mask]
            
            total_detections += len(dt_instances)
            
            # Write results
            if len(dt_instances) > 0:
                boxes_xyxy = dt_instances.boxes.numpy()
                track_ids = dt_instances.obj_idxes.numpy()
                write_results(
                    txt_path=predict_txt,
                    frame_id=(frame_id + 1),
                    bbox_xyxy=boxes_xyxy,
                    identities=track_ids + 1  # +1 as MOT benchmark requires positive
                )
        
        print(f"[INFO] Total {total_detections} detections written to {predict_txt}")


def main():
    parser = argparse.ArgumentParser("ByteTrack with LLM Filtering for RMOT")
    
    # ByteTrack paths
    parser.add_argument("--bytetrack_exp", type=str, 
                       default=str(Path.home() / "ByteTrack_LLM_dumb/exps/example/mot/yolox_x_mix_det.py"),
                       help="Path to ByteTrack experiment file")
    parser.add_argument("--bytetrack_ckpt", type=str,
                       default=str(Path.home() / "ByteTrack_LLM_dumb/pretrained/bytetrack_x_mot17.pth.tar"),
                       help="Path to ByteTrack checkpoint")
    
    # RMOT paths
    parser.add_argument("--rmot_path", type=str, required=True,
                       help="Path to RMOT dataset root")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    
    # Video/expression selection
    parser.add_argument("--video_ids", type=str, nargs="+", default=["0005", "0011", "0013"],
                       help="Video IDs to process (space separated)")
    parser.add_argument("--video_id", type=str, default=None,
                       help="Single video ID to process (overrides --video_ids)")
    parser.add_argument("--expression_json", type=str, default=None,
                       help="Single expression JSON file (if set, only processes this one)")
    
    # LLM settings
    parser.add_argument("--llm_api_url", type=str, default="http://localhost:11434/api/generate",
                       help="LLM API endpoint")
    parser.add_argument("--llm_model", type=str, default="qwen2.5vl",
                       help="LLM model name")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build list of sequences to process (same logic as inference.py)
    expressions_root = os.path.join(args.rmot_path, 'expression')
    seq_nums = []
    
    if args.video_id and args.expression_json:
        # Single video + single expression
        seq_nums = [[args.video_id, args.expression_json]]
    elif args.video_id:
        # Single video, all expressions
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, args.video_id)))
        seq_nums = [[args.video_id, ej] for ej in expression_jsons]
    else:
        # Multiple videos (default: 0005, 0011, 0013), all expressions
        for video_id in args.video_ids:
            expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
            for expression_json in expression_jsons:
                seq_nums.append([video_id, expression_json])
    
    print(f"[INFO] Processing {len(seq_nums)} sequences")
    
    # Initialize ByteTrack detector once (expensive operation)
    print(f"[INFO] Loading ByteTrack from {args.bytetrack_exp}")
    detector = ByteTrackDetector(
        exp_file=args.bytetrack_exp,
        ckpt_file=args.bytetrack_ckpt,
        device="gpu",
        fp16=False,
        fuse=True,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        fps=30
    )
    
    # Run inference on all sequences
    for seq_num in seq_nums:
        print(f"\n{'='*60}")
        print(f"[INFO] Evaluating seq {seq_num}")
        print(f"{'='*60}")
        
        inference = ByteTrackLLMInference(
            detector=detector,
            output_dir=args.output_dir,
            rmot_path=args.rmot_path,
            seq_num=seq_num,
            llm_api_url=args.llm_api_url,
            llm_model=args.llm_model
        )
        
        inference.detect()
        
        # Reset tracker for next sequence
        detector.reset()
        
        # Clear references
        del inference
        torch.cuda.empty_cache()
    
    print(f"\n[INFO] Done! Processed {len(seq_nums)} sequences.")


if __name__ == "__main__":
    main()
