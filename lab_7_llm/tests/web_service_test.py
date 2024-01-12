"""
Checks that E2E scenario allows to get desired metrics values
"""
# pylint: disable=duplicate-code
import unittest
from collections import namedtuple

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')
    TestClient = namedtuple('TestClient', 'post')

from lab_7_llm.service import app


class WebServiceTest(unittest.TestCase):
    """
    Tests tokenize function
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = app

        cls._client = TestClient(app)

    @pytest.mark.lab_7_llm
    @pytest.mark.mark10
    def test_e2e_ideal(self):
        """
        Ideal tokenize scenario
        """
        url = "/infer"
        input_text = "What is the capital of France?"

        payload = {"question": input_text}
        response = self._client.post(url, json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn('infer', response.json())
        print(response.json().get('infer'))
        self.assertIsNotNone(response.json().get('infer'))
