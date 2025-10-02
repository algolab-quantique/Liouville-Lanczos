"""
    Liouville-Lanczos: A library for Many-Body Green's function on quantum and classical computer.
    Copyright (C) 2024  Alexandre Foley
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
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .Lanczos_components import Logger
import time

class LoggerOp(Logger):
    def __init__(self):
        self.operators = []
        self.a = []
        self.b = []
        self.then = time.time()
    def __call__(self,iteration,recursion_operator:SparsePauliOp,a_i,b_i):
        now = time.time()
        delta = now - self.then
        self.then = now
        self.operators.append(recursion_operator)
        print(f"iteration: {iteration}\ntime delta: {delta}\n newest op size: {len(self.operators[iteration])}, a_{iteration}={a_i},b_{iteration}={b_i}")