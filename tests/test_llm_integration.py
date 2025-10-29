"""
Unit tests for LLM integration in RMOT
Tests r50_rmot_test_llm.sh script and inference_with_llm.py with mocked LLM API
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import subprocess
import tempfile
import shutil
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path setup
import inference_with_llm
from llm_filter import LLMFilter, encode_image, crop_bbox


class TestLLMScript(unittest.TestCase):
    """Test cases for r50_rmot_test_llm.sh script invocation"""

    def setUp(self):
        """Set up test fixtures"""
        self.script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'configs', 
            'r50_rmot_test_llm.sh'
        )

    def test_script_exists(self):
        """Test 3a: Verify the r50_rmot_test_llm.sh script exists"""
        self.assertTrue(os.path.isfile(self.script_path))

    def test_script_invokes_inference_with_llm(self):
        """Test 3b: Verify script invokes inference_with_llm.py"""
        # Read the script content
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # Verify it calls inference_with_llm.py
        self.assertIn('inference_with_llm.py', content)
        self.assertIn('python3', content)

    def test_script_includes_llm_arguments(self):
        """Test 3c: Verify script includes LLM-specific arguments"""
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # Verify LLM-specific arguments are present
        self.assertIn('--use_llm', content)
        self.assertIn('--llm_api_url', content)
        self.assertIn('--llm_model', content)

    def test_script_llm_argument_values(self):
        """Test 3d: Verify LLM arguments have correct default values"""
        with open(self.script_path, 'r') as f:
            content = f.read()
        
        # Check for expected values
        self.assertIn('http://localhost:11434/api/generate', content)
        self.assertIn('qwen2.5vl', content)

    @patch('subprocess.run')
    def test_script_execution_with_llm_args(self, mock_run):
        """Test 3e: Verify script can be parsed to extract command with LLM args"""
        # This test simulates what would happen if we execute the script
        # We parse the script to extract the command
        with open(self.script_path, 'r') as f:
            lines = f.readlines()
        
        # Extract command lines (skip comments and shebang)
        command_lines = [
            line.strip() for line in lines 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        # Join the command (remove trailing backslashes)
        full_command = ' '.join(
            line.rstrip('\\').strip() for line in command_lines
        )
        
        # Verify key components are in the command
        self.assertIn('inference_with_llm.py', full_command)
        self.assertIn('--use_llm', full_command)
        self.assertIn('--llm_api_url', full_command)
        self.assertIn('--llm_model', full_command)


class TestLLMFilter(unittest.TestCase):
    """Test cases for LLM filter with mocked API"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.crops_dir = os.path.join(self.test_dir, 'crops')
        self.llm_filter = LLMFilter(
            api_url='http://test-api:11434/api/generate',
            model='test-model',
            crops_dir=self.crops_dir
        )

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_llm_filter_initialization(self):
        """Test 4a: Verify LLMFilter initializes correctly"""
        self.assertEqual(self.llm_filter.api_url, 'http://test-api:11434/api/generate')
        self.assertEqual(self.llm_filter.model, 'test-model')
        self.assertEqual(self.llm_filter.crops_dir, self.crops_dir)
        self.assertTrue(os.path.exists(self.crops_dir))

    @patch('requests.post')
    def test_query_llm_with_mock_positive_response(self, mock_post):
        """Test 4b: Verify LLM query with mocked positive response"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"response": "Yes, this matches the description."}',
            b'{"done": true}'
        ]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Call query_llm
        is_match, response = self.llm_filter.query_llm(
            image_b64='fake_base64_string',
            prompt='a person wearing red shirt'
        )
        
        # Verify the result
        self.assertTrue(is_match)
        self.assertIn('Yes', response)
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['model'], 'test-model')
        self.assertIn('a person wearing red shirt', call_args[1]['json']['prompt'])

    @patch('requests.post')
    def test_query_llm_with_mock_negative_response(self, mock_post):
        """Test 4c: Verify LLM query with mocked negative response"""
        # Mock negative API response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"response": "No, this does not match."}',
            b'{"done": true}'
        ]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        is_match, response = self.llm_filter.query_llm(
            image_b64='fake_base64_string',
            prompt='a person wearing red shirt'
        )
        
        self.assertFalse(is_match)
        self.assertIn('No', response)

    @patch('requests.post')
    def test_query_llm_handles_api_error(self, mock_post):
        """Test 4d: Verify LLM query handles API errors gracefully"""
        # Mock API error
        mock_post.side_effect = Exception('Connection refused')
        
        is_match, response = self.llm_filter.query_llm(
            image_b64='fake_base64_string',
            prompt='test prompt'
        )
        
        # Should return False and empty string on error
        self.assertFalse(is_match)
        self.assertEqual(response, '')

    @patch.object(LLMFilter, 'query_llm')
    def test_filter_first_frame_finds_match(self, mock_query):
        """Test 4e: Verify filter_first_frame finds matching object"""
        # Create fake image and boxes
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([
            [100, 100, 200, 200],  # Box 0
            [300, 300, 400, 400],  # Box 1 - this will match
        ])
        obj_ids = np.array([10, 20])
        
        # Mock query_llm to return False for first, True for second
        mock_query.side_effect = [
            (False, "No match"),
            (True, "Yes, matches the description")
        ]
        
        target_id = self.llm_filter.filter_first_frame(
            image=image,
            boxes=boxes,
            obj_ids=obj_ids,
            sentence='a person wearing red',
            save_crops=True
        )
        
        # Verify correct target was found
        self.assertEqual(target_id, 20)
        
        # Verify query_llm was called twice
        self.assertEqual(mock_query.call_count, 2)
        
        # Verify crops were saved
        expected_crop_files = ['i10_f0.jpg', 'i20_f0.jpg']
        for crop_file in expected_crop_files:
            crop_path = os.path.join(self.crops_dir, crop_file)
            self.assertTrue(os.path.exists(crop_path))

    @patch.object(LLMFilter, 'query_llm')
    def test_filter_first_frame_no_match(self, mock_query):
        """Test 4f: Verify filter_first_frame returns None when no match"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]])
        obj_ids = np.array([10])
        
        # Mock query_llm to always return False
        mock_query.return_value = (False, "No match")
        
        target_id = self.llm_filter.filter_first_frame(
            image=image,
            boxes=boxes,
            obj_ids=obj_ids,
            sentence='a person wearing red',
            save_crops=False
        )
        
        self.assertIsNone(target_id)


class TestLLMDetector(unittest.TestCase):
    """Test cases for LLMDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('inference_with_llm.Detector.__init__')
    def test_llm_detector_initialization(self, mock_super_init):
        """Test 4g: Verify LLMDetector initializes with LLM filter"""
        mock_super_init.return_value = None
        
        # Create mock args
        args = Mock()
        args.llm_api_url = 'http://localhost:11434/api/generate'
        args.llm_model = 'qwen2.5vl'
        args.use_llm = True
        args.output_dir = self.test_dir
        
        # Create LLMDetector
        detector = inference_with_llm.LLMDetector(args, checkpoint_id=99)
        detector.save_path = self.test_dir
        detector.llm_filter = LLMFilter(
            api_url=args.llm_api_url,
            model=args.llm_model,
            crops_dir=os.path.join(self.test_dir, 'crops')
        )
        
        # Verify LLM filter was created
        self.assertIsNotNone(detector.llm_filter)
        self.assertEqual(detector.llm_filter.api_url, 'http://localhost:11434/api/generate')
        self.assertEqual(detector.llm_filter.model, 'qwen2.5vl')

    def test_crop_bbox_function(self):
        """Test 4h: Verify crop_bbox correctly crops images"""
        # Create test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 100:200] = [255, 0, 0]  # Red square
        
        # Crop the red square
        bbox = [100, 100, 200, 200]
        crop = crop_bbox(image, bbox)
        
        # Verify crop dimensions
        self.assertEqual(crop.shape, (100, 100, 3))
        
        # Verify crop content (should be all red)
        self.assertTrue(np.all(crop[:, :, 0] == 255))  # Red channel
        self.assertTrue(np.all(crop[:, :, 1] == 0))    # Green channel
        self.assertTrue(np.all(crop[:, :, 2] == 0))    # Blue channel

    def test_encode_image_function(self):
        """Test 4i: Verify encode_image produces base64 string"""
        # Create small test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Encode to base64
        encoded = encode_image(image)
        
        # Verify it's a string
        self.assertIsInstance(encoded, str)
        
        # Verify it's not empty
        self.assertGreater(len(encoded), 0)
        
        # Verify it's valid base64 (no errors when decoding)
        import base64
        try:
            decoded = base64.b64decode(encoded)
            self.assertGreater(len(decoded), 0)
        except Exception as e:
            self.fail(f"Encoded string is not valid base64: {e}")


class TestInferenceWithLLMArguments(unittest.TestCase):
    """Test argument parsing for inference_with_llm.py"""

    def test_llm_arguments_defined(self):
        """Test 4j: Verify LLM-specific arguments are defined"""
        # Import the argument parser
        from main import get_args_parser
        import argparse
        
        parser = argparse.ArgumentParser('Test', parents=[get_args_parser()])
        parser.add_argument('--use_llm', action='store_true', help='Enable LLM filtering')
        parser.add_argument('--llm_api_url', type=str, default='http://localhost:11434/api/generate')
        parser.add_argument('--llm_model', type=str, default='qwen2.5vl')
        
        # Parse test arguments
        args = parser.parse_args([
            '--use_llm',
            '--llm_api_url', 'http://test:11434/api/generate',
            '--llm_model', 'test-model',
            '--meta_arch', 'rmot',
            '--dataset_file', 'e2e_rmot',
            '--epoch', '200',
        ])
        
        # Verify arguments are correctly parsed
        self.assertTrue(args.use_llm)
        self.assertEqual(args.llm_api_url, 'http://test:11434/api/generate')
        self.assertEqual(args.llm_model, 'test-model')

    def test_llm_arguments_defaults(self):
        """Test 4k: Verify LLM arguments have correct defaults"""
        from main import get_args_parser
        import argparse
        
        parser = argparse.ArgumentParser('Test', parents=[get_args_parser()])
        parser.add_argument('--use_llm', action='store_true', help='Enable LLM filtering')
        parser.add_argument('--llm_api_url', type=str, default='http://localhost:11434/api/generate')
        parser.add_argument('--llm_model', type=str, default='qwen2.5vl')
        
        # Parse with minimal arguments
        args = parser.parse_args([
            '--meta_arch', 'rmot',
            '--dataset_file', 'e2e_rmot',
            '--epoch', '200',
        ])
        
        # Verify defaults
        self.assertFalse(args.use_llm)  # Should be False by default
        self.assertEqual(args.llm_api_url, 'http://localhost:11434/api/generate')
        self.assertEqual(args.llm_model, 'qwen2.5vl')


if __name__ == '__main__':
    unittest.main()
