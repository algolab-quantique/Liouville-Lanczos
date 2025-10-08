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

class Lanczos():
    """
    Modular implementation of Lanczos recursion and Lanczos-polynomial combined recursion.
    The vectors in the Krylov subspace must implement the multiplication with
    a scalar

    The Lanczos recursion, in the very abstract, construct an orthonormal basis
    spanning a Krylov subspace, whithin an Hilbert space. The Hilbert space 
    doesn't have to be the usual Hilbert space of quantum mecanic (spanned by 
    wavefunction or wavevectors), but any Hilbert space that may be of 
    interest. 
    
    For exemple to computation response function, one can can define an Hilbert
    spanned by quantum operators. This allow the computation of response 
    function at any temperature, provided one can compute averages at a given 
    temperature.

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
    def __init__(self,inner_product:Inner_product,liouvillian:Liouvillian,sum:Summation,epsilon=1e-13 ,logger = None):
        self.inner_prod  = inner_product
        self._logger = logger
        self.Liouvillian = liouvillian
        self.sum = sum
        self.epsilon = epsilon

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def setter(self,new_logger):
        self._logger = new_logger

    def polynomial_hybrid(self,H,f_0,other_vectors,max_k,min_b=1e-10):
        """
        Perform Lanczos recursion in the Krylov subspace spanned by the repeated 
        action of H on the initial vector f_0, and use it to compute polynomial 
        decomposition of the response functions with the list of other_vectors.
        Perform up to max_k iteration.
        The Krylov subspace is considered completely determined when the new 
        direction defined by the action of H on the previous vector has a norm 
        smaller then min_b.
        The action of H on f_0 is defined by the Liouvillian implementation 
        supplied to the Lanczos constructor.
        The norm of the vectors are defined by the inner_product supplied to the
        Lanczos contructor.
        Returns that a and b and mu coefficients of the Lanczos recursion.

        Concerning the a,b coefficient, consult the docstring of this class 
        __call__ method.
        There are as many set of mu coefficiant in output as there are vectors 
        in other_vectors. Along with the a and b coefficients, the mu
        coefficients contain the necessary information to compute the response
        function $i(v_i(t)|f_0(0))\\theta(t)$ where $v_i$ is a vector found in 
        other_vectors.
        Liouville-Lanczos' Green submodule contains the facilities to compute 
        the Fourrier transform of those response function.
        """
        i=0
        b = [np.sqrt(self.inner_prod(f_0,f_0,real_result=True,Name="b0"))]
        f_i = f_0/b[-1]
        multimoments = [[self.inner_prod(o,f_i,real_result=False,Name=f"m{m}_{0}") for m,o in enumerate(other_vectors)] ]
        f_ip = self.Liouvillian(-H,f_i)
        a_i = self.inner_prod(f_ip,f_i,real_result=True,Name="a0")
        if self.logger:
            self.logger(i,f_i,a_i,b[-1])
        f_ip = self.sum(f_ip, - a_i*f_i)
        b_ip = np.sqrt(self.inner_prod(f_ip,f_ip,real_result=True,Name="b1"))
        f_ip = f_ip / b_ip
        a = [a_i]
        f_i,f_im = f_ip,f_i
        for i in range(1,max_k):
            if b_ip < min_b:
                return a,b,multimoments
            multimoments.append([self.inner_prod(o,f_i,real_result=False,Name=f"m{m}_{i}") for m,o in enumerate(other_vectors)]) #**not** always real
            f_ip = self.Liouvillian(-H,f_i)
            try:
                a_i = self.inner_prod(f_ip,f_i,real_result=True,Name=f"a_{i}") #always real
                a.append(a_i)
            except Exception as e:
                print(f"anomalous termination a at iteration {i}")
                print(e)
                return a,b,multimoments
            b.append(b_ip)
            if self.logger:
                self.logger(i,f_i,a[-1],b[-1])
            f_ip = self.sum(f_ip,- a_i*f_i,- b[-1]*f_im)
            try:
                b2 = self.inner_prod(f_ip,f_ip,real_result=True,Name=f"b^2_{i+1}") #Always real
                assert b2>self.epsilon , f"b^2={b2} is smaller than {self.epsilon}, terminating"
                b_ip = np.sqrt(b2)
            except Exception as e:
                print(f"anomalous termination b at iteration {i}")
                print(e)
                return a,b,multimoments
            f_ip = f_ip / b_ip
            f_i,f_im = f_ip,f_i
        if self.logger:
            self.logger(i,f_i,a[-1],b[-1])
        return a,b,multimoments
         
    def __call__(self,H,f_0,max_k,min_b=1e-10):
        """
        Perform Lanczos recursion in the Krylov subspace spanned by the repeated 
        action of H on the initial vector f_0. Perform up to max_k iteration.
        The Krylov subspace is considered completely determined when the new 
        direction defined by the action of H on the previous vector has a norm 
        smaller then min_b.
        The action of H on f_0 is defined by the Liouvillian implementation 
        supplied to the Lanczos constructor.
        The norm of the vectors are defined by the inner_product supplied to the
        Lanczos contructor.
        Returns that a and b coefficients of the Lanczos recursion.

        The coefficient can be viewed as a tridiagonal matrix, with the a_0..a_n
        coefficients on the diagonnal and the b_1..b_n coeffient on the first 
        super- and sub-diagonnals. When assemble in that manner, it is the 
        matrix representation of the Liouvillian.
        Given f_0, this matrix form can be used to compute the response function
        $i(f_0(t)|f_0(0))\\theta(t)$ where the inner product $(a|b)$ is the one 
        provided to the constructor.
        Liouville Lanczos's Green submodule provide facilities to compute the 
        value of the Fourier transform of those response function from 
        the coefficiants.
        """
        a,b,_ = self.polynomial_hybrid(H,f_0,[],max_k,min_b)
        return a,b
    