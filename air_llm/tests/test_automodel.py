import sys
import unittest

sys.path.insert(0, "../airllm")

from airllm.auto_model import _apply_profile_defaults




class TestAutoModel(unittest.TestCase):
    def test_profile_defaults_should_not_override_explicit_values(self):
        kwargs = {
            'deployment_profile': 'rpi5_8gb_sd',
            'device': 'cpu',
            'prefetching': True,
        }
        out = _apply_profile_defaults(kwargs)
        self.assertEqual(out['device'], 'cpu')
        self.assertTrue(out['prefetching'])
        self.assertEqual(out['prefetch_window'], 1)

    def test_unknown_profile_should_raise(self):
        with self.assertRaises(ValueError):
            _apply_profile_defaults({'deployment_profile': 'unknown'})
