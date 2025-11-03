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
from .Lanczos_components import Inner_product,Liouvillian,Summation
class Matrix_Liouvillian(Liouvillian):
    def __call__(self,H,f):
        return H@f-f@H

class Matrix_Hamiltonian(Liouvillian):
    def __init__(self, E0,sign):
        self.E0 = E0
        self.sign = sign
    def __call__(self,H,f):
        return self.sign*(H-self.E0*np.eye(H.shape[0]))@f

class Matrix_sum(Summation):
    def __call__(self,*X):
        X = np.array(X)
        return np.add.reduce(X,axis=0)

class DensityMatrix_inner_product(Inner_product):
    """
    This operator inner product is defined as $\\mathrm{Tr}\\left[ \\rho \\{A , B^\dagger\\} \\right]$
    where $\\rho$ is a density matrix, $A$ and $B$ are operators, and the curly braces
    denote the anti-commutator.
    This class implements the computation with classical linear algebra (e.g. numpy).
    """
    def __init__(self,DensityMatrix):
        """
        initialise the inner product with the density matrix.
        """
        self.DM =DensityMatrix
    def __call__(self, A,B,*args,**kwargs) -> float:
        """
        compute the inner product between A and B.
        """
        Bd = np.conj(B.T)
        P = A@Bd
        T = Bd@A
        rP = self.DM@P
        rT = self.DM@T
        II = np.trace(  self.DM@(P + T) ) #most reliably precise so far. not that great.
        out = II
        return out
    
class Hamiltonian_inner_product(Inner_product):
    """
    Implement a statevector inner product.
    """
    def __call__(self,A,B):
        return  np.conj(A.T)@B
    
class MatrixState_inner_product(Inner_product):
    """
    This operator inner product is defined as $\\bra{\\psi} \\rho \\{A , B^\dagger\\} \\ket{\\psi}$
    where $\\ket{\\psi}$ is a quantum state, $A$ and $B$ are operators, and the 
    curly braces denote the anti-commutator.
    This class implements the computation with classical linear algebra (e.g. numpy).
    It is more efficiant than Density_Matrix_inner_product whenever the density 
    matrix is a projector.
    """
    def __init__(self,state):
        """
        initialize the inner product with the state.
        """
        self.state =state

    def __call__(self, A,B,*args,**kwargs) -> float:
        """
        compute the inner product between A and B.
        """
        Bd = np.conj(B.T)
        # f = A@Bd + Bd@A
        sa = A@self.state
        sa = Bd@sa
        sb = Bd@self.state
        sb = A@sb
        stated = np.conj(self.state)
        out = stated@(sa+sb)
        
        return out