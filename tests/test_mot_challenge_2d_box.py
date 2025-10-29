"""
Unit tests for MotChallenge2DBox dataset class
Tests ground truth file path construction and sequence ID handling
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TrackEval')))

from trackeval.datasets import MotChallenge2DBox


class TestMotChallenge2DBox(unittest.TestCase):
    """Test cases for MotChallenge2DBox dataset class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory structure
        self.test_dir = tempfile.mkdtemp()
        self.gt_folder = os.path.join(self.test_dir, 'gt')
        self.trackers_folder = os.path.join(self.test_dir, 'trackers')
        
        # Create directory structure
        os.makedirs(self.gt_folder, exist_ok=True)
        os.makedirs(self.trackers_folder, exist_ok=True)
        
        # Create seqmaps directory
        self.seqmaps_dir = os.path.join(self.gt_folder, 'seqmaps')
        os.makedirs(self.seqmaps_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_gt_fol_construction_with_split_folder(self):
        """Test 1: Verify gt_fol is correctly constructed when SKIP_SPLIT_FOL is False"""
        # Create necessary directory structure
        split_fol = 'MOT17-train'
        gt_split_path = os.path.join(self.gt_folder, split_fol)
        trackers_split_path = os.path.join(self.trackers_folder, split_fol)
        os.makedirs(gt_split_path, exist_ok=True)
        os.makedirs(trackers_split_path, exist_ok=True)
        
        # Create a dummy seqmap file
        seqmap_file = os.path.join(self.seqmaps_dir, 'MOT17-train.txt')
        with open(seqmap_file, 'w') as f:
            f.write('0005+001\n')
        
        # Create dummy GT file
        gt_file_dir = os.path.join(gt_split_path, '0005')
        os.makedirs(gt_file_dir, exist_ok=True)
        gt_file = os.path.join(gt_split_path, '0005', '001', 'gt.txt')
        os.makedirs(os.path.dirname(gt_file), exist_ok=True)
        with open(gt_file, 'w') as f:
            f.write('1,1,100,100,50,50,1,1,1\n')
        
        # Create dummy tracker file
        tracker_dir = os.path.join(trackers_split_path, 'test_tracker')
        os.makedirs(tracker_dir, exist_ok=True)
        tracker_file = os.path.join(tracker_dir, '0005', '001', 'predict.txt')
        os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
        with open(tracker_file, 'w') as f:
            f.write('1,1,100,100,50,50,1,1,1\n')
        
        config = {
            'GT_FOLDER': self.gt_folder,
            'TRACKERS_FOLDER': self.trackers_folder,
            'TRACKERS_TO_EVAL': ['test_tracker'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'SKIP_SPLIT_FOL': False,
            'GT_LOC_FORMAT': '{gt_folder}/{video_id}/{expression_id}/gt.txt',
            'SEQMAP_FOLDER': self.seqmaps_dir,
        }
        
        dataset = MotChallenge2DBox(config)
        
        # Verify gt_fol includes the split folder
        expected_gt_fol = os.path.join(self.gt_folder, 'MOT17-train')
        self.assertEqual(dataset.gt_fol, expected_gt_fol)
        
        # Verify tracker_fol includes the split folder
        expected_tracker_fol = os.path.join(self.trackers_folder, 'MOT17-train')
        self.assertEqual(dataset.tracker_fol, expected_tracker_fol)

    def test_gt_fol_construction_without_split_folder(self):
        """Test 1b: Verify gt_fol is correctly constructed when SKIP_SPLIT_FOL is True"""
        # Create a dummy seqmap file
        seqmap_file = os.path.join(self.seqmaps_dir, 'MOT17-train.txt')
        with open(seqmap_file, 'w') as f:
            f.write('0005+001\n')
        
        # Create dummy GT file directly in gt_folder
        gt_file = os.path.join(self.gt_folder, '0005', '001', 'gt.txt')
        os.makedirs(os.path.dirname(gt_file), exist_ok=True)
        with open(gt_file, 'w') as f:
            f.write('1,1,100,100,50,50,1,1,1\n')
        
        # Create dummy tracker file
        tracker_dir = os.path.join(self.trackers_folder, 'test_tracker')
        os.makedirs(tracker_dir, exist_ok=True)
        tracker_file = os.path.join(tracker_dir, '0005', '001', 'predict.txt')
        os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
        with open(tracker_file, 'w') as f:
            f.write('1,1,100,100,50,50,1,1,1\n')
        
        config = {
            'GT_FOLDER': self.gt_folder,
            'TRACKERS_FOLDER': self.trackers_folder,
            'TRACKERS_TO_EVAL': ['test_tracker'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'SKIP_SPLIT_FOL': True,
            'GT_LOC_FORMAT': '{gt_folder}/{video_id}/{expression_id}/gt.txt',
            'SEQMAP_FOLDER': self.seqmaps_dir,
        }
        
        dataset = MotChallenge2DBox(config)
        
        # Verify gt_fol does NOT include the split folder
        self.assertEqual(dataset.gt_fol, self.gt_folder)
        self.assertEqual(dataset.tracker_fol, self.trackers_folder)

    def test_gt_file_path_construction_with_sequence_ids(self):
        """Test 2: Verify GT file paths are correctly constructed using sequence IDs"""
        split_fol = 'MOT17-train'
        gt_split_path = os.path.join(self.gt_folder, split_fol)
        trackers_split_path = os.path.join(self.trackers_folder, split_fol)
        os.makedirs(gt_split_path, exist_ok=True)
        os.makedirs(trackers_split_path, exist_ok=True)
        
        # Create seqmap with multiple sequences
        seqmap_file = os.path.join(self.seqmaps_dir, 'MOT17-train.txt')
        sequences = ['0005+001', '0005+002', '0011+003']
        with open(seqmap_file, 'w') as f:
            for seq in sequences:
                f.write(f'{seq}\n')
        
        # Create GT files for each sequence
        for seq in sequences:
            video_id, expression_id = seq.split('+')
            gt_file = os.path.join(gt_split_path, video_id, expression_id, 'gt.txt')
            os.makedirs(os.path.dirname(gt_file), exist_ok=True)
            with open(gt_file, 'w') as f:
                f.write('1,1,100,100,50,50,1,1,1\n')
            
            # Create tracker files
            tracker_file = os.path.join(trackers_split_path, 'test_tracker', video_id, expression_id, 'predict.txt')
            os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
            with open(tracker_file, 'w') as f:
                f.write('1,1,100,100,50,50,1,1,1\n')
        
        config = {
            'GT_FOLDER': self.gt_folder,
            'TRACKERS_FOLDER': self.trackers_folder,
            'TRACKERS_TO_EVAL': ['test_tracker'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'SKIP_SPLIT_FOL': False,
            'GT_LOC_FORMAT': '{gt_folder}/{video_id}/{expression_id}/gt.txt',
            'SEQMAP_FOLDER': self.seqmaps_dir,
        }
        
        dataset = MotChallenge2DBox(config)
        
        # Verify sequences are correctly loaded
        self.assertEqual(len(dataset.seq_list), 3)
        self.assertIn('0005+001', dataset.seq_list)
        self.assertIn('0005+002', dataset.seq_list)
        self.assertIn('0011+003', dataset.seq_list)
        
        # Test GT file path construction for each sequence
        for seq in sequences:
            video_id, expression_id = seq.split('+')
            expected_path = os.path.join(
                dataset.gt_fol, 
                video_id, 
                expression_id, 
                'gt.txt'
            )
            # This is what the GT_LOC_FORMAT produces
            formatted_path = config['GT_LOC_FORMAT'].format(
                gt_folder=dataset.gt_fol,
                video_id=video_id,
                expression_id=expression_id
            )
            self.assertEqual(formatted_path, expected_path)
            self.assertTrue(os.path.isfile(expected_path))

    def test_sequence_id_parsing_in_load_raw_file(self):
        """Test 2b: Verify sequence IDs are correctly parsed in _load_raw_file"""
        split_fol = 'MOT17-train'
        gt_split_path = os.path.join(self.gt_folder, split_fol)
        trackers_split_path = os.path.join(self.trackers_folder, split_fol)
        os.makedirs(gt_split_path, exist_ok=True)
        os.makedirs(trackers_split_path, exist_ok=True)
        
        # Create seqmap
        seqmap_file = os.path.join(self.seqmaps_dir, 'MOT17-train.txt')
        with open(seqmap_file, 'w') as f:
            f.write('0005+001\n')
        
        # Create GT file with actual data
        video_id, expression_id = '0005', '001'
        gt_file = os.path.join(gt_split_path, video_id, expression_id, 'gt.txt')
        os.makedirs(os.path.dirname(gt_file), exist_ok=True)
        with open(gt_file, 'w') as f:
            # Frame, ID, x, y, w, h, conf, class, visibility
            f.write('1,1,100,100,50,50,1,1,1\n')
            f.write('2,1,105,105,50,50,1,1,1\n')
        
        # Create image directory to determine sequence length
        img_dir = os.path.join(gt_split_path, video_id)
        os.makedirs(img_dir, exist_ok=True)
        # Create dummy image files
        for i in range(2):
            img_file = os.path.join(img_dir, f'{i:06d}.jpg')
            with open(img_file, 'w') as f:
                f.write('')
        
        # Create tracker files
        tracker_file = os.path.join(trackers_split_path, 'test_tracker', video_id, expression_id, 'predict.txt')
        os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
        with open(tracker_file, 'w') as f:
            f.write('1,1,100,100,50,50,0.9,1,1\n')
            f.write('2,1,105,105,50,50,0.9,1,1\n')
        
        config = {
            'GT_FOLDER': self.gt_folder,
            'TRACKERS_FOLDER': self.trackers_folder,
            'TRACKERS_TO_EVAL': ['test_tracker'],
            'BENCHMARK': 'MOT17',
            'SPLIT_TO_EVAL': 'train',
            'SKIP_SPLIT_FOL': False,
            'GT_LOC_FORMAT': '{gt_folder}/{video_id}/{expression_id}/gt.txt',
            'SEQMAP_FOLDER': self.seqmaps_dir,
        }
        
        dataset = MotChallenge2DBox(config)
        
        # Test loading raw GT file with sequence ID parsing
        seq = '0005+001'
        raw_data = dataset._load_raw_file(tracker='test_tracker', seq=seq, is_gt=True)
        
        # Verify the data was loaded correctly
        self.assertEqual(raw_data['seq'], seq)
        self.assertEqual(raw_data['num_timesteps'], 2)
        self.assertIsNotNone(raw_data['gt_ids'])
        self.assertEqual(len(raw_data['gt_ids']), 2)


if __name__ == '__main__':
    unittest.main()
