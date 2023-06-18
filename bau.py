#!/usr/bin/env python3.11

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

def main():
    parser = argparse.ArgumentParser(description='Try out new LLMs more quickly')
    parser.add_argument('-a','--arg', help='Help for arg', required=False)
    parser.add_argument('--license', action='store_true', help='Display the license information')
    args = parser.parse_args()

    if args.license:
        print_license()
        return

    if args.arg:
        print(args.arg)
        return args.arg

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
