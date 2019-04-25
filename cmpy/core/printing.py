# -*- coding: utf-8 -*-
"""
Created on 14 Oct 2018
@author: Dylan Jones

project: cmpy
version: 1.0

Printing utilities
==================

"""
import time
import sys
import shutil
from threading import Thread
import numpy as np


class Symbols:

    pm = "\u00B1"

    Delta = "\u0394"
    Lambda = "\u039B"

    delta = "\u03B4"
    lamb = "\u03BB"
    xi = "\u03BE"


def format_num(num, unit="b", div=1024):
    for scale in ['','k','M','G','T','P','E','Z']:
        if abs(num) < div:
            return f"{num:.1f} {scale}{unit}"
        num /= div
    return f"{num:.1f} Z{unit}"


def short_time_str(secs):
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


def matrix_string(array, element_width=None, col_header=None, row_header=None):
    if element_width is None:
        w = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                w = max(w, len(str(array[i, j])))
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
        string += " "
        string += " ".join([f"{h:^{w}}" for h in col_header]) + "\n"

    for i in range(array.shape[0]):
        line = "[" if row_header is None else f"{row_header[i]:{w_h}}["
        for j in range(array.shape[1]):
            val = array[i, j]
            print(val)
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


class ConsoleLine:

    def __init__(self, header=None, width=None):
        self._header = "" if header is None else header + ": "
        self._width = None
        self.set_width(width)

    def set_width(self, width=None):
        self._width = shutil.get_terminal_size((80, 20))[0] - 1 if width is None else width

    def write(self, txt):
        txt = self._format(txt)
        sys.stdout.write("\r" + txt)
        sys.stdout.flush()

    def end(self, txt=""):
        if txt:
            txt = self._format(txt)
            sys.stdout.write("\r" + txt + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\n")

    def _format(self, txt):
        if not isinstance(txt, str):
            txt = str(txt)
        txt = self._header + txt
        if len(txt) > self._width - 1:
            txt = txt[:self._width-3] + "..."
        else:
            txt = f"{txt: <{self._width}}"
        return txt

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.end()


class MovingAverage:

    def __init__(self, size=100):
        self._i = 0
        self.mem = np.zeros(size)

    def update(self, value):
        self._i += 1
        self.mem = np.roll(self.mem, 1)
        self.mem[0] = value
        return self.get()

    def _get_memory(self):
        return self.mem if self._i > self.mem.shape[0] else self.mem[:self._i]

    def get(self):
        mem = self._get_memory()
        return np.mean(mem, axis=0) if len(mem) > 0 else 0


class Timer:

    def __init__(self, n=100):
        self._t_start = 0
        self._t_prev = 0

        self._t_iter = MovingAverage(n)
        self._t_total = 0
        self.start()

    @property
    def t_total(self):
        return self._t_total

    @property
    def t_iter(self):
        return self._t_iter.get()

    @property
    def memory(self):
        return self._t_iter.mem

    def start(self):
        self._t_start = time.perf_counter()
        self._t_prev = self._t_start

    def update(self):
        t = time.perf_counter()
        self._t_total = t - self._t_start
        self._t_iter.update(t - self._t_prev)
        self._t_prev = t

    def t_left(self, n_left):
        return n_left * self.t_iter

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update()

    def __str__(self):
        self.update()
        line = "-" * 25
        result = f"Time: {time_str(self.t_total)}"
        return f"\n{result}\n{line}"


class Progress(ConsoleLine):

    def __init__(self, iterable=None, total=None, header=None, unit="it", f=5, enabled=True):
        self._enabled = enabled
        super().__init__(header)
        self._iterable = iterable
        self._n = total if total else len(iterable)
        self._timer = Timer(10)
        self._update_freq = f
        self._thread_running = False
        self._thread = None

        self._i = 0
        self._txt = ""
        self._unit = unit
        self._scale, self._scalestr = self._get_scale(self._n)[:2]
        if f is not None:
            self._start_callback()
        self._start()

    def __iter__(self):
        self._start()
        for obj in self._iterable:
            yield obj
            self.update()
        self.end()

    def set_description(self, txt=None):
        if txt is not None:
            self._txt = txt

    def set_progress(self, i):
        self._i = i
        if self._thread is None:
            self._update_output()

    def update(self, txt=None, increment=1):
        self._i += increment
        self.set_description(txt)
        self._timer.update()

        if self._thread is None:
            self._update_output()

    def _update_output(self):
        if not self._enabled:
            return
        info = self._format_progress(self._i)
        if self._txt:
            info += f" {self._txt}"
        info += " [" + self._format_counter(self._i)
        info += ", " + self._format_time(self._i) + "]"
        self.set_width()
        self.write(info)

    def _start(self):
        self._i = 0
        self._timer.start()

    def _start_callback(self):
        self._thread_running = True
        self._thread = Thread(target=self._callback)
        self._thread.start()

    def _callback(self):
        while self._thread_running:
            self._update_output()
            time.sleep(1/self._update_freq)

    def end(self, *args):
        if self._thread is not None:
            self._thread_running = False
            self._thread.join()
        if not self._enabled:
            return
        self._update_output()
        super().end()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *args, **kwargs):
        self.end()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    # ==============================================================================================

    def _format_progress(self, i, dec=0):
        progress = 100 * i / self._n
        return f"{progress:>3.{dec}f}%"

    def _format_counter(self, i):
        count = int(i / self._scale)
        total = int(self._n / self._scale)
        return f"{count}{self._scalestr}/{total}{self._scalestr}"

    def _format_time(self, i):
        t_total = short_time_str(self._timer.t_total)
        t_eta = short_time_str(self._timer.t_left(self._n-i))
        if self._timer.t_iter == 0:
            it_per_sec = 0
        else:
            it_per_sec = 1/self._timer.t_iter
        it_str = self._scale_format(it_per_sec)
        return f"{t_total}<{t_eta}, {it_str}"

    @staticmethod
    def _get_scale(n):
        dec = 1
        scale = 1
        scalestr = ""
        if (10 < n) and (n < 5000):
            dec = 0
        if n >= 5000:
            scale = 1000
            scalestr = "k"
        elif n >= 1e6:
            scale = 1e6
            scalestr = "M"
        return scale, scalestr, dec

    def _scale_format(self, n):
        scale, scalestr, dec = self._get_scale(n)
        if n < 0.5:
            n *= 60
            unit = self._unit + "/min"
        else:
            unit = self._unit + "/s"
        return f"{n/scale:.{dec}f}{scalestr} {unit}"


def prange(*args, **kwargs):
    return Progress(range(*args), **kwargs)
