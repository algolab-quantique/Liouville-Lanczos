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

from abc import ABC,abstractmethod
from typing import Any, Optional
from numbers import Complex

class Vector(ABC):
    """
    Stand in only for the type hints.
    In concrete implementation, replace with the concretly used type.
    Your concrete type does not need to inherit from this class.
    But it must support multiplication with scalar values.
    In practice, those vectors could be quantum operators.
    """
    @abstractmethod
    def __mul__(self,other:Complex)->"Vector":
        ...


class Inner_product(ABC):
    """
    Interface class for an inner_product definition for use in Lanczos algorithm 
    implementation of Liouville-Lanczos. Must accept two "vectors" and return a 
    (maybe complex) scalar. 
    """
    @abstractmethod
    def __call__(self,A:Vector,B:Vector,real_result:bool,Name:Optional[str])->Complex:
        """
        compute the inner product between the vector A and B.
        If a real result is expected, Lanczos will set the real_result boolean 
        to True. This information can be used to perform some optimizations.
        Lanczos will supply a Name string, useful when the inner_product is a
        quantum computer task.
        """
        ...

class Liouvillian(ABC):
    """
    Interface class for the Liouvillian implementation.
    The call implements the desired action of a quantum object on a Krylov 
    subspace vector. If the call method implement the commutator, the first 
    argument to the Liouvillian is a Hamiltonian, the second a quantum operator,
    then the proper quantum Liouvillian is implemented.
    """
    @abstractmethod
    def __call__(self,Hamiltonian,vector:Vector)->Vector:
        ...

class Summation(ABC):
    """
    Interface class for the Summation implementation
    """
    @abstractmethod
    def __call__(self,*Summands:list[Vector])->Vector:
        """
        perform the sum of the supplied Krylov subspace vectors, return the 
        resulting vector.
        """
        ...