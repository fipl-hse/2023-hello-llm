"""
Checks that datasets gets analyzed proreprly
"""
# pylint: disable=duplicate-code, assignment-from-no-return
import unittest
from pathlib import Path

import pytest

from config.lab_settings import LabSettings
from config.reference_scores import ReferenceAnalysisScores
from lab_7_llm.main import RawDataImporter, RawDataPreprocessor
from lab_7_llm.start import main


class DatasetWorkingTest(unittest.TestCase):
    """
    Tests analyse function
    """

    @pytest.mark.lab_7_llm
    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    def test_analyze_ideal(self):
        """
        Ideal analyze scenario
        """
        main()

        settings = LabSettings(Path(__file__).parent.parent / 'settings.json')

        importer = RawDataImporter(settings.parameters.dataset)
        importer.obtain()

        preprocessor = RawDataPreprocessor(importer.raw_data)

        dataset_analysis = preprocessor.analyze()

        references = ReferenceAnalysisScores()

        self.assertEqual(dataset_analysis,
                         references.get(settings.parameters.dataset))

        self.assertEqual(len(dataset_analysis), 6)
