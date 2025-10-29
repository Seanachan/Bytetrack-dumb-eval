"""
Evaluation script for ByteTrack+LLM results using TrackEval
"""
import sys
import os
from pathlib import Path

# Add TrackEval to path
TRACKEVAL_ROOT = Path(__file__).parent / "TrackEval"
sys.path.insert(0, str(TRACKEVAL_ROOT))

from trackeval import Evaluator, get_metrics, get_datasets


def evaluate_bytetrack_results(results_dir, gt_base_path, dataset_name="mot_challenge_2d_box"):
    """
    Evaluate ByteTrack+LLM results using TrackEval
    
    Args:
        results_dir: Directory containing predict.txt files
                    Structure: {results_dir}/{video_id}/{expression_id}/predict.txt
        gt_base_path: Base path to ground truth files
                     Structure: {gt_base_path}/labels_with_ids/image_02/{video_id}/
        dataset_name: Dataset type for evaluation
    """
    
    # Setup evaluator
    evaluator = Evaluator(dataset_config={
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': False,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': '/tmp/trackeval_errors.txt',
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_PROGRESS_BAR': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_METRICS': True,
        'OUTPUT_PER_SEQUENCE': True,
        'PLOT_CURVES': False,
    })
    
    # Get dataset
    dataset_list = [get_datasets([dataset_name])[0]({
        'GT_FOLDER': gt_base_path,
        'TRACKERS_FOLDER': results_dir,
        'SKIP_SPLIT_FOL': True,
    })]
    
    # Get metrics
    metrics_list = ['HOTA', 'CLEAR', 'Identity']
    metrics = [get_metrics([m])[0]() for m in metrics_list]
    
    # Run evaluation
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(output_msg)
    
    return output_res


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Evaluate ByteTrack+LLM Results")
    parser.add_argument("--results_dir", type=str, default="./outputs/bytetrack_results",
                       help="Directory with tracking results")
    parser.add_argument("--gt_base_path", type=str, default="/home/seanachan/RMOT/KITTI/training",
                       help="Base path to ground truth")
    parser.add_argument("--dataset", type=str, default="mot_challenge_2d_box",
                       help="Dataset type for evaluation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"[ERROR] Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    print(f"[INFO] Evaluating results from: {args.results_dir}")
    print(f"[INFO] Using ground truth from: {args.gt_base_path}")
    
    evaluate_bytetrack_results(
        results_dir=args.results_dir,
        gt_base_path=args.gt_base_path,
        dataset_name=args.dataset
    )
