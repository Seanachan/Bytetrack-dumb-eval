"""
LLM-based target filtering for RMOT
Adapted from ByteTrack_LLM_dumb approach
"""
import os
import requests
import base64
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional


def encode_image(image_array: np.ndarray) -> str:
    """Encode numpy image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def crop_bbox(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """Crop image using bounding box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


class LLMFilter:
    def __init__(self, api_url: str = "http://localhost:11434/api/generate", 
                 model: str = "qwen2.5vl",
                 crops_dir: str = "crops"):
        self.api_url = api_url
        self.model = model
        self.crops_dir = crops_dir
        os.makedirs(crops_dir, exist_ok=True)
        
    def query_llm(self, image_b64: str, prompt: str) -> Tuple[bool, str]:
        """
        Query LLM with an image and prompt
        Returns: (is_match: bool, response_text: str)
        """
        payload = {
            "model": self.model,
            "prompt": f"{prompt}\nConfirm whether it meets the requirements. If yes, answer 'yes'. If not, answer 'no'.",
            "images": [image_b64]
        }
        
        try:
            resp = requests.post(self.api_url, json=payload, stream=True, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            return False, ""
        
        result_text = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        result_text += data["response"]
                except:
                    continue
        
        is_match = "yes" in result_text.lower()
        return is_match, result_text
    
    def filter_first_frame(self, image: np.ndarray, boxes: np.ndarray, 
                          obj_ids: np.ndarray, sentence: str, 
                          save_crops: bool = True) -> Optional[int]:
        """
        Filter detections in first frame using LLM
        
        Args:
            image: Original image (H, W, 3)
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            obj_ids: Object IDs [N]
            sentence: Natural language query/expression
            save_crops: Whether to save crop images to disk
            
        Returns:
            target_id: ID of the matched object, or None if no match
        """
        print(f"[LLM Filter] Processing {len(boxes)} detections with prompt: '{sentence}'")
        
        for idx, (box, obj_id) in enumerate(zip(boxes, obj_ids)):
            crop = crop_bbox(image, box)
            
            if save_crops:
                crop_path = os.path.join(self.crops_dir, f"i{obj_id}_f0.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            # Encode and query LLM
            crop_b64 = encode_image(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            is_match, response = self.query_llm(crop_b64, sentence)
            
            print(f"[LLM Filter] ({idx+1}/{len(boxes)}) ID={obj_id}: {response.strip()} -> {'MATCH' if is_match else 'NO MATCH'}")
            
            if is_match:
                print(f"[LLM Filter] ✓ Target found: ID={obj_id}")
                return int(obj_id)
        
        print(f"[LLM Filter] ✗ No target matched the description")
        return None
    
    def filter_instances(self, image: np.ndarray, dt_instances, 
                        sentence: str, frame_idx: int = 0):
        """
        Filter Instances object to keep only LLM-matched targets
        
        Args:
            image: Original image
            dt_instances: Instances object with boxes, obj_idxes, scores, etc.
            sentence: Natural language query
            frame_idx: Current frame index (0 for first frame)
            
        Returns:
            filtered_instances: Instances with only matched objects
        """
        if frame_idx > 0:
            # Only filter on first frame
            return dt_instances
        
        target_id = self.filter_first_frame(
            image, 
            dt_instances.boxes.numpy(),
            dt_instances.obj_idxes.numpy(),
            sentence,
            save_crops=True
        )
        
        if target_id is None:
            # No match found, return empty instances
            return dt_instances[np.array([], dtype=bool)]
        
        # Keep only the matched target
        keep_mask = dt_instances.obj_idxes == target_id
        return dt_instances[keep_mask.numpy()]
