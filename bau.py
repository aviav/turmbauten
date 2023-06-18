"""
Turmbauten
Copyright (C) 2023 Tobias Fankhänel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import configparser

def main(args=None):
    parser = argparse.ArgumentParser(description='Try out new LLMs more quickly')

    config = configparser.ConfigParser()
    config.read('config.ini')

    parser.add_argument('-a','--arg', default=config.get('DEFAULT', 'ArgDefaultValue', fallback=None), help='Help for arg')
    parser.add_argument('--license', action='store_true', help='Display the license information')
    parser.add_argument('--config', help='Path to the configuration file', default='config.ini')

    args = parser.parse_args(args)

    if args.license:
        print_license()
        return {'arg': args.arg, 'license': True}

    return {'arg': args.arg, 'license': False}

def print_license():
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """)

if __name__ == "__main__":
    main()
