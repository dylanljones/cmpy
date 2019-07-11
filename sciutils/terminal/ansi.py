# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import sys
from .core import STDOUT, IS_WIN, is_terminal

ESC = "\033["
ENABLED = False
WARN = False


def encode(cmd):
    return ESC + str(cmd)


def _send(cmd):
    string = encode(cmd)
    sys.__stdout__.write(string)
    sys.__stdout__.flush()


class Cursor:

    @staticmethod
    def set(x, y, mode="H"):
        """ Move cursor to screen location v, h

        Parameters
        ----------
        x: int
        y: int
        mode: str, optional
            allowed modes: "H", "f", default "H"
        """
        _send(f"{x};{y}{mode}")

    @staticmethod
    def set_column(x):
        _send(f"{x}G")

    @staticmethod
    def upper_left(mode="H"):
        """ Move cursor to upper left corner

        Parameters
        ----------
        mode: str, optional
            allowed modes: "H", "f", ";H", ";f", default "H"
        """
        # Move cursor to upper left corner
        _send(f"{mode}")

    @staticmethod
    def up(n=1):
        """ Move cursor up n times

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}A")

    @staticmethod
    def down(n):
        """ Move cursor down n times

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}B")

    @staticmethod
    def right(n):
        """ Move cursor right n times

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}C")

    @staticmethod
    def left(n):
        """ Move cursor left n times

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}D")

    @staticmethod
    def next_line(n=1):
        """ Move cursor to beginning of line n lines down

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}E")

    @staticmethod
    def prev_line(n=1):
        """ Move cursor to beginning of line n lines up

        Parameters
        ----------
        n: int, optional
            default: 1
        """
        _send(f"{n}F")


def clear_line(mode="all"):
    """ Clears the current line of the console

    Parameters
    ----------
    mode: str, optional
        direction to clear, valid arguments:
        "left", "right", "all
        default: "all"
    """
    modes = dict(right=0, left=1, all=2)
    _send(f"{modes[mode]}K")


def clear_screen(mode="all"):
    """ Clears the terminal screen

    Parameters
    ----------
    mode: str, optional
        direction to clear, valid arguments:
        "up", "down", "all
        default: "all"
    """
    modes = dict(down=0, up=2, all=2)
    _send(f"{modes[mode]}J")


def _enable_win(std_id=STDOUT):
    """ Enable Windows 10 cmd.exe ANSI VT Virtual Terminal Processing.

    Parameters
    ----------
    std_id: int, default: STDOUT
        std-handle index

    Returns
    -------
    success: bool
    """
    from ctypes import byref, POINTER, windll, WINFUNCTYPE
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    kernel = windll.kernel32
    get_std_handle = WINFUNCTYPE(HANDLE, DWORD)(('GetStdHandle', kernel))
    get_file_type = WINFUNCTYPE(DWORD, HANDLE)(('GetFileType', kernel))
    get_console_mode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(('GetConsoleMode', kernel))
    set_console_mode = WINFUNCTYPE(BOOL, HANDLE, DWORD)(('SetConsoleMode', kernel))

    if std_id == 1:
        # stdout
        h = get_std_handle(-11)
    elif std_id == 2:
        # stderr
        h = get_std_handle(-12)
    else:
        return False

    if h is None or h == HANDLE(-1):
        return False

    file_type_char = 0x0002
    if (get_file_type(h) & 3) != file_type_char:
        return False

    mode = DWORD()
    if not get_console_mode(h, byref(mode)):
        return False

    enable_virtual_terminal = 0x0004
    if (mode.value & enable_virtual_terminal) == 0:
        set_console_mode(h, mode.value | enable_virtual_terminal)
    return True


def enable_ansi(std_id=STDOUT, warn=False):
    """ Enable ANSI escape sequences

    Parameters
    ----------
    std_id: int, default: STDOUT
        std-handle index
    warn: bool
        Warn when not in console if True
    """
    if not ENABLED:
        # WINDOWS
        if IS_WIN:
            _enable_win(std_id)
            if not is_terminal() and warn:
                msg = "You're being piped! Ansi escape sequences may not work!"
                print("[WARNING] " + msg)
                print()
        else:
            raise OSError("Only windows supported")


enable_ansi(warn=WARN)
