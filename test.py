import unittest
from unittest.mock import patch
from io import StringIO
from bau import main, print_license

class TestYourScript(unittest.TestCase):
    def test_main(self):
        @patch('sys.stdout', new_callable=StringIO)
        def test_license(self, mock_stdout):
            expected_license = print_license()
            main(['--license'])
            self.assertEqual(mock_stdout.getvalue().strip(), expected_license.strip())

if __name__ == "__main__":
    unittest.main()
