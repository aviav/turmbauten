import unittest
from unittest.mock import patch
import bau

class TestMain(unittest.TestCase):
    def test_license(self):
        with patch('builtins.print') as mock_print:
            result = bau.main(['--license'])
            mock_print.assert_called_once_with('\n    This program is free software: you can redistribute it and/or modify\n    it under the terms of the GNU General Public License as published by\n    the Free Software Foundation, either version 3 of the License, or\n    (at your option) any later version.\n\n    This program is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n    GNU General Public License for more details.\n\n    You should have received a copy of the GNU General Public License\n    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n    ')
            self.assertEqual(result, {'arg': 'default value', 'license': True})

    def test_arg(self):
        result = bau.main(['-a', 'test_arg'])
        self.assertEqual(result, {'arg': 'test_arg', 'license': False})

if __name__ == '__main__':
    unittest.main()
