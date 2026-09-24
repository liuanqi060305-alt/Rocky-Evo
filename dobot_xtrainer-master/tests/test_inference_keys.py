import unittest

from experiments.run_inference_openpi import select_key_input


class InferenceKeySelectionTests(unittest.TestCase):
    def test_terminal_y_is_not_overwritten_by_opencv_residual_key(self):
        self.assertEqual(
            select_key_input("y", ord("f"), accepted_keys={"y", "f"}),
            "y",
        )

    def test_invalid_terminal_key_does_not_hide_valid_opencv_y(self):
        self.assertEqual(
            select_key_input("x", ord("y"), accepted_keys={"y", "f"}),
            "y",
        )

    def test_unlisted_safety_key_is_ignored(self):
        self.assertIsNone(
            select_key_input("r", 255, accepted_keys={"y", "f"})
        )

    def test_xyz_override_key_still_works(self):
        self.assertEqual(
            select_key_input(None, ord("I"), accepted_keys={"i", "f"}),
            "i",
        )

    def test_invalid_terminal_key_does_not_hide_window_control_key(self):
        self.assertEqual(
            select_key_input("x", ord("r"), accepted_keys={"r", "q"}),
            "r",
        )

    def test_opencv_no_key_value_is_ignored(self):
        self.assertIsNone(
            select_key_input(None, -1, accepted_keys={"y", "f"})
        )


if __name__ == "__main__":
    unittest.main()
