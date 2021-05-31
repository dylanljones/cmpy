# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones


def mit_header(name: str, year: str):
    s = f"""
    Copyright (c) {year}, {name}

    This code is licensed under the MIT License. The copyright notice in LICENSE.txt
    file in the root directory and this permission notice shall be included in all
    copies or substantial portions of the Software.
    """
    return s


def remove_header_lines(lines):
    index = 0
    start_header = -1
    end_header = 0
    while index < len(lines):
        line = lines[index].strip()
        comment_line = line.startswith("#")
        if start_header < 0 and comment_line:
            start_header = index
        elif start_header >= 0 and not comment_line:
            end_header = index - 1
            break
        index += 1
    num_lines = end_header - start_header + 1
    print(start_header, end_header, num_lines)


def replace_file_header(filename: str, header: str):
    # Read file
    with open(filename, "r") as fh:
        lines = fh.readlines()
    remove_header_lines(lines)
    # Write file
    with open(filename, "w") as fh:
        fh.writelines(lines)


def main():
    replace_file_header("_project.py", "")


if __name__ == "__main__":
    main()
