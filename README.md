# Introduction

The software library is a python implentation of methods for the computation of Many-Body
Green's function based on the Liouvillian formalism, along with methods for the manipulation of Green's function. The Liouvillian formalism applied to the computation of Green's function is very flexible. As such, the present library currently offer an numpy based ED backend, a qiskit based quantum computer backend and a quimb based matrix product operator backend.

The quantum computer implementation offers, in principle, an exponential advantage over ED.
This statement is strongly depedent on the means to produce the steady state of the target system, a QMA-hard problem left to the user. Indeed, while no general **and** efficient method exist to prepare a steady state solution, efficient specialized methods and heuristics are possible.

The time and memory complexity of the quantum-classical hybrid algorithm is still exponential in the number of Green's function coefficients to be computed. 
A fully quantum implementation is possible, a would have a polynomial time complexity (upcoming).

With current hardware, the fully classical algorithms are faster and more accurate for almost all problems.

# Installation

1. Download the repo
2. Open a terminal and move to the folder containing the project
3. Emit the command ``` pip install ./ ``` in the project folder