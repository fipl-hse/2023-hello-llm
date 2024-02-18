"""
Checks the inference of the model
"""
# pylint: disable=R0801
import unittest
from pathlib import Path

import pytest

from lab_7_llm.tests.infer_sample_test import run_model_inference_check
from lab_8_llm.main import LLMPipeline


class ModelWorkingTest(unittest.TestCase):
    """
    Tests analyse function
    """

    @pytest.mark.lab_8_llm
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_analyze_ideal(self):
        """
        Ideal inference scenario
        """
        self.assertIsNone(run_model_inference_check(Path(__file__).parent.parent, LLMPipeline))
