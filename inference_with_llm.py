"""
RMOT Inference with LLM Filtering
Based on inference.py but adds LLM-based target identification
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from inference import *
from llm_filter import LLMFilter


class LLMDetector(Detector):
    """Extended Detector class with LLM filtering capability"""
    
    def __init__(self, args, checkpoint_id=None, model=None, seq_num=2):
        super().__init__(args, checkpoint_id, model, seq_num)
        
        # Initialize LLM filter
        self.llm_filter = LLMFilter(
            api_url=getattr(args, 'llm_api_url', 'http://localhost:11434/api/generate'),
            model=getattr(args, 'llm_model', 'qwen2.5vl'),
            crops_dir=os.path.join(self.save_path, 'crops')
        )
        self.use_llm = getattr(args, 'use_llm', True)
        self.target_id = None  # Will be set by LLM on first frame
        
    def detect(self, prob_threshold=0.7, area_threshold=100):
        """
        Detection loop with LLM filtering
        """
        last_dt_embedding = None
        total_dts = 0
        total_occlusion_dts = 0
        print('Results are saved into {}'.format(self.save_path))
        if self.use_llm:
            print('[LLM] LLM filtering enabled')
        else:
            print('[LLM] LLM filtering disabled')

        track_instances = None
        loader = DataLoader(ListImgDataset(self.img_list), 1, num_workers=2)
        
        for i, (cur_img, ori_img) in enumerate(tqdm(loader)):
            cur_img, ori_img = cur_img[0], ori_img[0]

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            # Run TransRMOT model inference
            res = self.detr.inference_single_image(cur_img.cuda().float(), self.sentence, (seq_h, seq_w),
                                                   track_instances)
            track_instances = res['track_instances']

            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            dt_instances = track_instances.to(torch.device('cpu'))

            # Standard filtering
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
            dt_instances = self.filter_dt_by_ref_scores(dt_instances, 0.5)

            # LLM filtering on first frame
            if self.use_llm and i == 0 and len(dt_instances) > 0:
                print(f"\n[LLM] First frame: {len(dt_instances)} detections before LLM filter")
                
                # Query LLM to find target
                self.target_id = self.llm_filter.filter_first_frame(
                    image=ori_img.numpy(),
                    boxes=dt_instances.boxes.numpy(),
                    obj_ids=dt_instances.obj_idxes.numpy(),
                    sentence=self.sentence[0],
                    save_crops=True
                )
                
                if self.target_id is not None:
                    print(f"[LLM] Target locked: ID={self.target_id}\n")
                else:
                    print(f"[LLM] WARNING: No target matched! Tracking all objects.\n")
            
            # Filter to keep only target if LLM found one
            if self.use_llm and self.target_id is not None:
                keep_mask = dt_instances.obj_idxes == self.target_id
                dt_instances = dt_instances[keep_mask]

            num_occlusion = (dt_instances.labels == 1).sum()
            dt_instances.scores[dt_instances.labels == 1] *= -1
            total_dts += len(dt_instances)
            total_occlusion_dts += num_occlusion

            if args.visualization:
                cur_vis_img_path = os.path.join(self.save_path, 'frame_{}.jpg'.format(i))
                gt_boxes = None
                self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, gt_boxes=gt_boxes)

            tracker_outputs = self.tr_tracker.update(dt_instances)

            self.write_results(txt_path=os.path.join(self.save_path, 'predict.txt'),
                               frame_id=(i + 1),
                               bbox_xyxy=tracker_outputs[:, :4],
                               identities=tracker_outputs[:, 5])
                               
        gt_path = os.path.join(self.save_path, 'gt.txt')
        self.write_gt(gt_path, self.json_path,
                      os.path.join(args.rmot_path, 'KITTI/labels_with_ids/image_02', self.seq_num[0]), seq_h, seq_w)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    
    # Add LLM-specific arguments
    parser.add_argument('--use_llm', action='store_true', help='Enable LLM filtering')
    parser.add_argument('--llm_api_url', type=str, default='http://localhost:11434/api/generate',
                        help='LLM API endpoint')
    parser.add_argument('--llm_model', type=str, default='qwen2.5vl',
                        help='LLM model name')
    
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    checkpoint_id = int(args.resume.split('/')[-1].split('.')[0].split('t')[-1])
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    expressions_root = os.path.join(args.rmot_path, 'expression')
    video_ids = ['0005', '0011', '0013']

    seq_nums = []
    for video_id in video_ids:
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
        for expression_json in expression_jsons:
            seq_nums.append([video_id, expression_json])

    for seq_num in seq_nums:
        print('Evaluating seq {}'.format(seq_num))
        det = LLMDetector(args, checkpoint_id, model=detr, seq_num=seq_num)
        det.detect()
