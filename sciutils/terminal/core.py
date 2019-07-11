# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import os
import sys
import re
import shutil
import numpy as np

STDOUT = 1
STDERR = 2

IS_WIN = os.name == "nt"

RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")  # re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class Symbols:
    pm = "\u00B1"

    Delta = "\u0394"
    Lambda = "\u039B"

    delta = "\u03B4"
    lamb = "\u03BB"
    xi = "\u03BE"


def ansi_len(string):
    """ Get the length of ansi escape sequences in the given string

    Parameters
    ----------
    string: str
        Input text

    Returns
    -------
    ansi_len: int
    """
    return len(string) - raw_len(string)


def remove_ansi(string):
    """ Remove all ansi escape sequences from the given string

    Parameters
    ----------
    string: str
        Input text

    Returns
    -------
    cleared_str: str
    """
    return RE_ANSI.sub("", string)


def raw_len(string):
    """ Get the length of the given string without ansi escape sequences

    Parameters
    ----------
    string: str
        Input text

    Returns
    -------
    raw_len: int
    """
    return len(remove_ansi(string))


def argstr(args):
    """ Join arguments to a single string

    Parameters
    ----------
    args: array_like
        List of arguments

    Returns
    -------
    argstr: str
    """
    return " ".join([str(x) for x in args])


def trunc_string(string, width):
    """ Truncate the given text to a certain length

    Parameters
    ----------
    string: str
        Input text
    width: int
        width of truncated string

    Returns
    -------
    string: str
    """
    return string[:width - 3] + "..." if len(string) > width else string


def pad_string(string, width, orientation="<", char=" "):
    """ Pad the given text to a certain length

    Parameters
    ----------
    string: str
        Input text
    width: int
        width of padded string
    orientation: str, default: left
        Orientation of the text
    char: str
        Fill character for padding

    Returns
    -------
    string: str
    """
    return f"{string:{char}{orientation}{width}}"


def format_line(string, width, orient="<", char=" "):
    """ Pads or cuts string to given length

    Parameters
    ----------
    string: str
        Input text
    width: int
        Width of formatted line
    orient: str, default: left
        Orientation of the text if padded
    char: str
        Fill character for padding

    Returns
    -------
    line_str: str
    """
    width += ansi_len(string)
    return pad_string(trunc_string(string, width), width, orient, char)


def format_num(num, unit="", div=1000, dec=1):
    """ Convert number to formatted text with scale and unit

    Parameters
    ----------
    num: float
        number to format
    unit: str, optional
        unit of number
    div: int, default: 1000
        divisor of scale
    dec: int, default: 1
        number of decimal points
    Returns
    -------
    formatted_num: str
    """
    for scale in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < div:
            return f"{num:.{dec}f}{scale} {unit}"
        num /= div
    return f"{num:.{dec}f} {unit}"


def get_scale(num, div=1000):
    """ Gets the scale and it's string for a given number

    Parameters
    ----------
    num: float
        number to format
    div: int, default: 1000
        divisor of scale

    Returns
    -------
    scale: int
    scalestr: str
    """
    scale = 1
    for scalestr in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < div:
            break
        scale *= div
        num /= scale
    return scale, scalestr


def short_time_str(secs):
    """ Return a short formatted string for the given time

    Parameters
    ----------
    secs: float
        time value to format

    Returns
    -------
    time_str: str
    """
    if secs > 0:
        mins, secs = divmod(secs, 60)
        if mins > 60:
            hours, mins = divmod(mins, 60)
            return f"{hours:02.0f}:{mins:02.0f}h"
        else:
            return f"{mins:02.0f}:{secs:02.0f}"

    else:
        return "00:00"


def time_str(secs):
    """ Return a formatted string for the given time

    Parameters
    ----------
    secs: float
        time value to format

    Returns
    -------
    time_str: str
    """
    if secs < 1e-3:
        nanos = 1e6 * secs
        return f"{nanos:.0f} \u03BCs"
    elif secs < 1:
        millis = 1000 * secs
        return f"{millis:.1f} ms"
    elif secs < 60:
        return f"{secs:.1f} s"
    else:
        mins, secs = divmod(secs, 60)
        if mins < 60:
            return f"{mins:.0f}:{secs:04.1f} min"
        else:
            hours, mins = divmod(mins, 60)
            return f"{hours:.0f}:{mins:02.0f}:{secs:02.0f} h"


def matrix_string(matrix, element_width=None, col_header=None, row_header=None):
    """ Return a formatted strong of the given matrix

    Parameters
    ----------
    matrix: array_like
        Input matrix
    element_width: int, default: None
        Width for elements of matrix, if not specified maximal width of elements is used
    col_header: array_like of str, default: None
        List of optional collumn headers of the matrix
    row_header: array_like of str, default: None
        List of optional row headers of the matrix
    Returns
    -------
    matrix_str: str
    """
    if element_width is None:
        w = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                w = max(w, len(str(matrix[i, j])))
    else:
        w = element_width
    if row_header is not None:
        w_h = max([len(x) for x in row_header]) + 1
    else:
        w_h = 0

    string = ""
    if col_header is not None:
        if w_h:
            string += " " * w_h
        string += "" + " ".join([f"{h:^{w}}" for h in col_header]) + "\n"
    for i in range(matrix.shape[0]):
        line = "[" if row_header is None else f"{row_header[i]:{w_h}}["
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val == 0:
                s = "  "
            elif np.imag(val) == 0:
                s = str(np.real(val))
            elif np.real(val) == 0:
                s = str(np.imag(val)) + "j"
            else:
                s = str(val)
            line += f"{s:^{w}} "
        string += line[:-1] + "]\n"
    return string[:-1]


class Stdout:

    @staticmethod
    def write(s):
        sys.__stdout__.write(s)

    @staticmethod
    def flush():
        sys.__stdout__.flush()

    def overwrite_stdout(self):
        sys.stdout = self

    @staticmethod
    def reset_stdout():
        sys.stdout = sys.__stdout__


def is_terminal():
    """ bool: Checks if output is in console """
    return sys.stdout.isatty()


def terminal_size(width=100, height=20):
    """ Returns the usable size of the current console window

    Parameters
    ----------
    width: int, default: 100
        Default width if size couldn't be read
    height: int, default: 20
        Default height if size couldn't be read

    Returns
    -------
    size: tuple of int
    """
    t_width, t_height = tuple(shutil.get_terminal_size((width, height)))
    if t_width == 0:
        t_width = width
    if t_height == 0:
        t_height = width
    return t_width, t_height


def clear_screen():
    """ Clears the console screen """
    if is_terminal():
        if IS_WIN:
            cmd = "cls"
        else:
            cmd = "clear"
        os.system(cmd)


def line_input(prompt=""):
    """ Get input line from console

    Parameters
    ----------
    prompt: str, default: None
        Prompt text for input

    Returns
    -------
    input: str
    """
    if prompt:
        print(prompt)
    string = ""
    char = sys.stdin.read(1)
    while char:
        if char == "\n":
            break
        string += char
        char = sys.stdin.read(1)
    return string
