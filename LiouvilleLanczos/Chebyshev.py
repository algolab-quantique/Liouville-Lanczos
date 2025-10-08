
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
from Lanczos_components import Inner_product,Liouvillian,Summation


class triple_product:
    """
    class for the computation of the triple product
    $T_{ijk} = int_a^b f_i(x)f_j(x)f_k(x)d\mu(x)$
    where $\beta_{i+1}f_{i+1}(x) = (x-\alpha_i)f_i(x)-\beta_i f_{i-1}(x)$
    It can be shown that 
    $\beta_{i+1}T_{i+1,jk} = \beta_{j+1}T_{i,j+1,k} +  (\alpha_j-\alpha_i)T_{ijk} + \beta_j T_{i,j-1,k} * \beta_i T_{i-1,jk}$
    We obeserve that any to element have the same value if their index are a permutation of each other.
    $\int_a^bf_i(x)f_j(x)d\mu(x) = \delta_{ij}$ implies that $T_{ij0} = \delta_ij$
    and the polynomial nature of the $f_i$ implies that $T_{ijk}=0 \forall k>i+j$
    Because $f_i(x) = 0 \forall i<0$, $T_{ijk} = 0$ if any of $i,j$ or $k$ are less than 0.
    Tchebyshev polynomial have a very simple product structure, most of those formula can be greatly simplified in their case.
    """

class Chebyshev():
    """
    Implementation of the Tchebyshev recursion for the computation of Spectral function
    or Green function. From a computationnal standpoint, the only difference is 
    that the recursion coefficient are fixed by the chosen basis. Only the polynomial
    are to be computed and outputed. 
    """
    def __init__(self,inner_product:Inner_product,liouvillian:Liouvillian,sum:Summation, logger = None):
        """
        Much like the Lanczos method, require an implementation of an Inner_product,
        a Liouvillian and Summation.


        The constructor require callables implementing the following modular parts:
        Inner_product:  Takes two object (typically operators or vectors in the 
                        Hilbert space of the problem) and computes the inner 
                        product between the two, returning the scalar value. This 
                        inner product defines the Hilbert space this Lanczos 
                        implementation works in. The vectors of that Hilbert space
                        the Lanczos recursion works in are those objects.
                        are the objects the innerproduct takes for arguments.
        Liouvillian:    The generator for the Krylov subspace. Implements the 
                        operation between the Hamiltonian and the vector's 
                        spanning the Krylov subspace. Should be a commutator when
                        the inner product is on quantum operator, 
                        or the matrix product if the inner product is on states.
                        It act on a vector in the Hilbert space and 
                        return a different vector of the same Hilbert space. 
                        Strictly speaking, the Liouvillian us a super-operator, is
                        the commutator of the hamiltonian with an operator.
        Summation:      Sum an arbitrary number of vector in the algorithm's 
                        working Hilbert space.
   
        If one wanted to implement Lanczos recursion on wavefunction, then 
        the inner product would be the regular inner product between wavevectors,
        the Liouvillian class would implement matrix-vector product, and the
        summation would sum an arbitrary number of wavevectors
   
        The necessary interface is specified by the abstract classes in the type hints.
        Remember, python's type hints are just that: hints. You are not bound to use
        those abstract types, so long as you replicate their interface.
        """
        self.inner_prod  = inner_product
        self._logger = logger
        self.Liouvillian = liouvillian
        self.sum = sum

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def setter(self,new_logger):
        self._logger = new_logger
        
    def __call__(self,H,f_0,max_k,other_f=None,min_b=1e-12):
        """
        Perform Techbyshev recursion in the Krylov subspace spanned by the repeated 
        action of H on the initial vector f_0, and use it to compute polynomial 
        decomposition of the response functions with the list of other_f and itself.
        Perform up to max_k iteration.
        The action of H on f_0 is defined by the Liouvillian implementation 
        supplied to the constructor.
        The norm and overlap of the vectors are defined by the inner_product supplied
        to the contructor.
        Returns mu coefficients of the Tchebyshev polynomial decomposition.
        """
        i=0
        #f_{i+1} = 2Lf_i - f_{i-1}
        #mu_i = (f_i,f_0)
        #\im(G(\omega)) = 0 \forall |\omega| > 1
        if other_f ==None:
            mu = np.zeros(max_k)
        else:
            mu = np.zeros((max_k,len(other_f)))
        f_i = f_0
        if other_f == None:
            mu[i] = self.inner_prod(f_0,f_i)
        else:
            mu[i,0] = self.inner_prod(f_0,f_i)
            mu[i,1:] = [self.inner_prod(f_n,f_i) for f_n in other_f]
        f_ip = self.Liouvillian(H,f_i)
        if self.logger:
            self.logger(i,f_i,mu[i])
        f_ip = self.sum(2*f_ip, -1*f_i)
        f_i,f_im = f_ip,f_i
        for i in range(1,max_k):
            if other_f == None:
                mu[i] = self.inner_prod(f_0,f_i)
            else:
                mu[i,0] = self.inner_prod(f_0,f_i)
                mu[i,1:] = [self.inner_prod(f_n,f_i) for f_n in other_f]
            if self.logger:
                self.logger.log(i,f_i,mu[i])
            f_ip = self.Liouvillian(H,f_i)
            f_ip = self.sum(2*f_ip,- 1*f_im)
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger.log(i,f_i,mu_i)
        return mu_i