# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import time
from threading import Thread
import numpy as np
from sciutils.misc import Timer, MovingAverage
from .core import Stdout, is_terminal, terminal_size, clear_screen
from .core import raw_len, argstr, short_time_str, format_num, get_scale
from . import ansi

UPDATE_CHAR = "\r"


class Terminal(Stdout):

    def __init__(self, enabled=True, clear=False):
        self.enabled = enabled
        self.piped = not is_terminal()
        self.pos = np.zeros(2, dtype="int")
        self._delta = np.zeros(2, dtype="int")
        self.size = np.asarray(terminal_size(), dtype="int")
        self.cursor = ansi.Cursor()
        if clear:
            self.clear()

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def write_raw(self, string):
        super().write(str(string))

    # =========================================================================

    def _update_pos(self, string):
        if not string:
            return

        lines = string.splitlines()
        delta_y = string.count("\n")
        if delta_y:
            self.pos[0] = 0
        delta_x = raw_len(lines[-1])
        self.pos[0] += delta_x
        self.pos[1] += delta_y

    def write(self, *args):
        string = argstr(args)
        if self.enabled:
            self._update_pos(string)
            super().write(string)
            self.flush()

    def writeln(self, string=""):
        self.write(string + "\n")

    def updateln(self, *args):
        self.clearln()
        self.write(*args)

    @staticmethod
    def clear():
        clear_screen()

    def linestart(self):
        if not self.piped:
            self.cursor.set_column(0)
            self.pos[0] = 0

    def clearln(self):
        if self.piped:
            self.write_raw(UPDATE_CHAR + " " * self.width + UPDATE_CHAR)
        else:
            ansi.clear_line()
            self.cursor.set_column(0)
        self.pos[0] = 0


class MaTimer(Timer):

    def __init__(self, n=10, weights=None):
        super().__init__()
        self._ma = MovingAverage(n, weights)
        self.t_iter = 0
        self.t_prev = 0
        self.t_total = None

        self.start()

    @property
    def t(self):
        return self.seconds

    def start(self):
        super().start()
        self.t_prev = self.t

    def stop(self):
        self.t_total = self.seconds

    def update(self):
        t = self.t
        self.t_iter = self._ma.update(t - self.t_prev)
        self.t_prev = t

    def eta(self, n):
        return n * self.t_iter


class Progress(Terminal):

    def __init__(self, iterable=None, total=None, header=None, unit="It", f=5, enabled=True):
        super().__init__(enabled, False)
        self._iterable = iterable
        self._n = None
        self._i = 0

        self._timer = MaTimer(10)

        self._txt = ""
        self.header = ""
        self.unit = ""
        self.scale = 1
        self.scalestr = ""

        self._thread = None
        self._update_freq = None
        self._thread_running = False

        # Setup
        self.setup(iterable, total, header, unit, f)

    def setup(self, iterable=None, total=None, header=None, unit="It", f=5):
        # Check if iteration number is given by either an iterable or as an integer
        self._iterable = None
        self._n = None
        if iterable:
            self._iterable = iterable
            self._n = len(iterable)
        elif total:
            self._n = total

        # Set current iteration to beginning
        self._i = 0

        # Set up the header, unit and scale of the output line
        self._txt = ""
        self.header = "" if header is None else header
        self.unit = unit
        if self._n:
            self.scale, self.scalestr = get_scale(self._n)

        # Set up variables for the output-thread
        self._update_freq = f
        self._thread_running = False
        self._thread = None

        # If frequncy f specified, start thread
        if f is not None:
            self.start_callback_thread()

    def start_callback_thread(self):
        self._thread_running = True
        self._thread = Thread(target=self._callback)
        self._thread.start()

    def stop_callback_thread(self):
        self._thread_running = False
        self._thread.join()

    @property
    def manual_update(self):
        return self._thread is None

    @property
    def open_mode(self):
        return (self._n is None) and (self._iterable is None)

    @property
    def progress(self):
        return self._i / self._n

    @property
    def time(self):
        return self._timer.seconds

    @property
    def n_left(self):
        return self._n - self._i

    def enable(self):
        super().enable()
        self.start_callback_thread()

    def disable(self):
        super().disable()
        self.stop_callback_thread()

    # =========================================================================

    # Output parts

    def _header_text(self):
        if self.header and self._txt:
            string = self.header + ": " + self._txt
        elif self.header and not self._txt:
            string = self.header
        elif not self.header and self._txt:
            string = self._txt
        else:
            string = ""
        return string

    def _progress_str(self, dec=0):
        progress = 100 * self.progress
        return f"{progress:>3.{dec}f}%"

    def _counter_str_open(self):
        return format_num(self._i, dec=0)

    def _counter_str(self):
        count = int(self._i / self.scale)
        total = int(self._n / self.scale)
        return f"{count}{self.scalestr}/{total}{self.scalestr}"

    def _total_time_str(self):
        return short_time_str(self.time)

    def _eta_time_str(self):
        return short_time_str(self._timer.eta(self._n-self._i))

    def _iter_time_str(self):
        t_iter = self._timer.t_iter
        it_ps = 0 if t_iter == 0 else 1 / t_iter
        if it_ps < 0.5:
            t_scale = "min"
            it_ps *= 60
        else:
            t_scale = "s"
        return format_num(it_ps, unit=f"{self.unit}/{t_scale}", dec=1)

    # Main mechanics

    def _output(self):
        string = self._header_text()
        string += f" {self._progress_str()}"
        string += " ["
        string += f"{self._counter_str()}"
        string += f", {self._total_time_str()}<{self._eta_time_str()}"
        string += f", {self._iter_time_str()}"
        string += "]"
        return string

    def _open_output(self):
        string = self._header_text()
        string += f" {self._counter_str_open()}"
        string += " ["
        string += f"{self._total_time_str()}"
        string += f", {self._iter_time_str()}"
        string += "]"
        return string

    def _update_output(self):
        if not self.enabled:
            return
        string = self._open_output() if self.open_mode else self._output()
        self.clearln()
        self.write(string)

    # =========================================================================

    def start(self):
        self._i = 0
        self._timer.start()

    def end(self):
        if self._thread is not None:
            self.stop_callback_thread()
        if not self.enabled:
            return
        self._update_output()
        self.writeln()

    def set_description(self, txt=None):
        if txt is not None:
            self._txt = txt

    def set_idx(self, i):
        self._i = i
        if self.manual_update:
            self._update_output()

    def update(self, txt=None, incr=1):
        self._timer.update()
        self._i += incr
        self.set_description(txt)

        if self.manual_update:
            self._update_output()

    def _callback(self):
        while self._thread_running:
            self._update_output()
            time.sleep(1/self._update_freq)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.end()

    def __iter__(self):
        self.start()
        for obj in self._iterable:
            yield obj
            self.update()
        self.end()


def prange(*args, **kwargs):
    return Progress(range(*args), **kwargs)
