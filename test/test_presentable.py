import io
import os
import sys
import unittest

from presentable import confusion_matrix
from unittest.mock import patch

from .data import reference_strings


class Test_sklearn_docs_example(unittest.TestCase):
    """
    Test presentable functionality.
    """

    @classmethod
    def setUpClass(self):
        self.sklearn_example_true = [2, 0, 2, 2, 0, 1]
        self.sklearn_example_pred = [0, 0, 2, 2, 0, 2]

        self.str_test_true = ["cat", "dog", "cat", "dog", "cat", "dog"]
        self.str_test_pred = ["cat", "cat", "dog", "dog", "dog", "dog"]

        self.small_true = [1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1]
        self.small_pred = [2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1]

        self.large_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.large_pred = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    def test_string_labels(self):
        """
        Labels should also work with strings.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.str_test_true,
                self.str_test_pred,
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.STR_TEST_OUTPUT,
            )

    def test_large_table(self):
        """
        Table with 10 rows/columns.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.large_true,
                self.large_pred,
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.LARGE_OUTPUT,
            )

    def test_small_table(self):
        """
        Example with only two labels.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.small_true,
                self.small_pred,
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.SMALL_OUTPUT,
            )

    def test_sklearn_example(self):
        """
        Reproduce the example from the sklearn docs.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.sklearn_example_true,
                self.sklearn_example_pred,
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.SKLEARN_EXAMPLE_OUTPUT,
            )

    def test_tabulate_customization(self):
        """
        Reproduce the example from the sklearn docs.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.sklearn_example_true,
                self.sklearn_example_pred,
                {"tablefmt": "fancy_outline"},
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.FANCY_TABLE_OUTPUT,
            )

    def test_full_customization(self):
        """
        Reproduce the example from the sklearn docs with custom args to sklearn and tabulate.
        """
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            confusion_matrix(
                self.sklearn_example_true,
                self.sklearn_example_pred,
                tabulate_args={
                    "tablefmt": "github",
                    "floatfmt": ".2f",
                },
                sklearn_args={"normalize": "all"},
            )
            self.assertEqual(
                mock_stdout.getvalue(),
                reference_strings.FULL_CUSTOMIZATION_OUTPUT,
            )


if __name__ == "__main__":
    unittest.main()
