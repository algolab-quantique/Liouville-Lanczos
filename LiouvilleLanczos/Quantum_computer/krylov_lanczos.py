#%%
from qkip.layout import QubitsLayout
from qkip.krylov import KrylovBasis, HeisenbergQKD
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimatorV2
from LiouvilleLanczos.Quantum_computer.QC_lanczos import relative_simplify_spo, separate_imag
import numpy as np
from LiouvilleLanczos.Lanczos_components import Inner_product as Base_inner_product
from typing import Optional
# %%


class krylov_inner_product_spo(Base_inner_product):
    def __init__(self,
            qkd: HeisenbergQKD,
            estimator: BaseEstimatorV2, 
            epsilon: int,
    ):
        self.qkd = qkd
        self.estimator = estimator
        self.eps = epsilon
    def __call__(self, A: SparsePauliOp, B: SparsePauliOp, real_result: bool=False, Name: Optional[str]=None):
        #do nothing with name for now
        Bc = B.adjoint()
        f = A@Bc+Bc@A
        f = relative_simplify_spo(f,self.eps)
        O_real, O_img = separate_imag(f)
        ans = self.qkd.estimate_on_gs(O_real, self.estimator)

        if(not real_result):
            return ans + 1j * self.qkd.estimate_on_gs(O_img, self.estimator)
        
        return ans