import autograd.numpy as np
import sys
import copy
import time
from typing import Callable, List, Tuple, Optional, Any
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

# 假设 QNSTR, QNSTR_LCP1, get_two_stage_LCP 已经实现并可导入
from QNSTRsolver import QNSTR
from QNSTR_LCP import QNSTRLCP
from generate_two_stage_LCP import get_two_SLCP

def proj(x: np.ndarray, domain: List[np.ndarray]) -> np.ndarray:
    """Projection onto a box constraint."""
    return np.maximum(domain[0], np.minimum(x, domain[1]))

class PSNASolver:
    """
    Parallel Two-Stage LCP Solver using QNSTR.
    """

    def __init__(
        self,
        func_f: List[Any],
        constant_f: np.ndarray,
        func_g: List[Any],
        domain_f: List[np.ndarray],
        domain_g: List[List[np.ndarray]],
        verbose: bool = True
    ):
        """
        Args:
            func_f: First stage function or [A, B, q] for linear.
            func_g: List of second stage functions or [T, M, q_] for linear.
            domain_f: [lower, upper] for x.
            domain_g: List of [lower, upper] for y_i.
            verbose: Whether to print progress.
        """
        self.func_f = func_f
        self.constant_f = constant_f 
        self.func_g = func_g
        self.domain_f = domain_f
        self.domain_g = domain_g
        self.verbose = verbose

    @staticmethod
    def is_linear_funcf(func_f: Any) -> bool:
        """Check if func_f is linear."""
        return isinstance(func_f, list)       
        
    @staticmethod
    def is_linear_funcg(func_g: List[Any]) -> bool:
        """Check if func_g is a list of linear blocks."""
        return all(isinstance(item, list) for item in func_g)

    @staticmethod
    def residual(x: np.ndarray, y: np.ndarray, func_f: List[Any], constant_f: np.ndarray, domain_f: List[np.ndarray]) -> float:
        """Compute residual for first stage."""
        if isinstance(func_f, list):
            F = func_f[0] @ x + func_f[1] @ y + constant_f
        else:
            F = func_f(x, y) + constant_f
        return np.linalg.norm(x - np.clip(x - F, domain_f[0], domain_f[1]))
   

    def Jacobian_fd(self, x: np.ndarray, y: np.ndarray, func_f: Callable):
        """Numerical Jacobian of A(x)"""
        epsilon = 1e-12
        dim = x.shape[0]
        I = np.eye(dim)
        Jaco = np.zeros_like(I)
        Fk = func_f(x, y)
        for i in range(x.shape[0]):
            x1 = x.copy()
            x1 += epsilon*I[:,i:i+1]
            F1 = func_f(x1, y)
            Jaco[:,i:i+1] = ((F1 - Fk)) / epsilon
        return Jaco  

    def JacobianY_fd(self, x: np.ndarray, y: np.ndarray, index_l: int, index_u:int, func_f: Callable):
        """Numerical Jacobian of A(x)"""
        epsilon = 1e-12
        dim = y.shape[0]
        I = np.eye(dim)
        Fk = func_f(x, y)
        dim1 = Fk.shape[0]
        Jaco = np.zeros([dim1,index_u-index_l])
        for i in range(index_u-index_l):
            y1 = y.copy()
            y1 += epsilon*I[:,index_l+i:index_l+i+1]
            F1 = func_f(x, y1)
            Jaco[:,i:i+1] = ((F1 - Fk)) / epsilon
        return Jaco     
    

    def update_parameter_parallel(
        self,
        y: np.ndarray,
        x: np.ndarray,
        domain: List[np.ndarray],
        lr: float,
        i: int,
        func: Any,
        epsilon: float = 1e-12,
        max_step: int = 1000
    ) -> np.ndarray:
        """Update y block in parallel."""
        if self.verbose:
            print(f'Updating subproblem {i+1}')
        subproblem = QNSTR(func, domain)
        y_, lr = subproblem.run(y, x, epsilon=epsilon, lr=lr, max_step=max_step, display=False)
        return y_, lr

    def update_parameter_parallel_linear_block(
        self,
        y: np.ndarray,
        x: np.ndarray,
        domain: List[List[np.ndarray]],
        lr: float,
        s: Tuple[int, int],
        func: List[List[Any]],
        epsilon: float = 1e-12,
        max_step: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update y block for linear case in parallel."""
        if self.verbose:
            print(f'Updating block {s[0]} ~ {s[1]}')
        N = len(func)
        M, q, N_mat = func[0][1], func[0][2], func[0][0]
        ldomainl, ldomainu = domain[0][0], domain[0][1]
        for i in range(1, N):
            M = block_diag(M, func[i][1])
            q = np.concatenate([q, func[i][2]], 0)
            N_mat = np.concatenate([N_mat, func[i][0]], 0)
            ldomainl = np.concatenate([ldomainl, domain[i][0]], 0)
            ldomainu = np.concatenate([ldomainu, domain[i][1]], 0)
        q = N_mat @ x + q
        ldomain = [ldomainl, ldomainu]
        #lfunc = [M, q]
        subproblem = QNSTRLCP(M, q, ldomain)
        y_ = subproblem.run(y, lr=1., epsilon=epsilon, max_step=max_step, display=False)
        """update Jacobian."""
        y_size_group = y_.shape[0]
        ind = ((np.abs(y_-ldomainl) - 1e-12 >= 0)*(np.abs(y_-ldomainu)-1e-12>=0))[:, 0]
        row1 = np.arange(0, y_size_group, 1)
        col1 = np.arange(0, y_size_group, 1)
        ones = np.ones(y_size_group)
        Ix = coo_matrix((ones, (row1, col1)), shape=(y_size_group, y_size_group))
        Di = coo_matrix((ind, (row1, col1)), shape=(y_size_group, y_size_group))
        try:
            Vi = (np.linalg.inv(Ix - Di + Di @ M) @ Di @ N_mat)
        except Exception as e:
            print(f"Matrix inversion failed in block {i}: {e}")
            Vi = np.zeros_like(M)
        Vi = np.asarray(Vi)
      
        return y_, Vi


    def stage1_QNSTR(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr_list: np.ndarray,
        group_size: int,
        y_size: int,
        num_cores: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: update y in parallel."""
        loop_index = range(0, len(lr_list), 1)
        results = Parallel(n_jobs=num_cores)(
            delayed(self.update_parameter_parallel)(
                y[y_size *i:y_size *(i+1),:],
                x,
                self.domain_g[i],
                lr_list[i, 0],
                i,
                self.func_g[i]
            ) for i in loop_index
        )
        for i in loop_index:
            y[y_size*i:y_size*(i+1)] = results[i][0]    
            lr_list[i,0] = results[i][1]            
        return y, lr_list    
    
    
    def stage1_QNSTR_LCP(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: List[int],
        lr_list: np.ndarray,
        group_size: int,
        y_size: int,
        num_cores: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: update y in parallel for linear LCP."""
        loop_index = range(len(index) - 1)
        results = Parallel(n_jobs=num_cores)(
            delayed(self.update_parameter_parallel_linear_block)(
                y[y_size * index[i]:y_size * index[i + 1], :],
                x,
                self.domain_g[index[i]:index[i + 1]],
                lr_list[i, 0],
                (index[i], index[i + 1]),
                self.func_g[index[i]:index[i + 1]]
            ) for i in loop_index
        )
        V = 0
        for i in loop_index:
            y[y_size * index[i]:y_size * index[i + 1]] = results[i][0]
            Bi = self.func_f[1][:, y_size * index[i]:y_size * index[i + 1]]
            Vi = Bi @ results[i][1]
            V = V + Vi      
        return y, -V
    
    
    def stage1_QNSTR_LCP1(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: List[int],
        lr_list: np.ndarray,
        group_size: int,
        y_size: int,
        num_cores: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: update y in parallel for linear LCP."""
        loop_index = range(len(index) - 1)
        results = Parallel(n_jobs=num_cores)(
            delayed(self.update_parameter_parallel_linear_block)(
                y[y_size * index[i]:y_size * index[i + 1], :],
                x,
                self.domain_g[index[i]:index[i + 1]],
                lr_list[i, 0],
                (index[i], index[i + 1]),
                self.func_g[index[i]:index[i + 1]]
            ) for i in loop_index
        )
        """update Jacobian."""
        V = 0
        for i in loop_index:
            y[y_size * index[i]:y_size * index[i + 1]] = results[i][0]
            y_ = np.zeros_like(y)
            y_[y_size * index[i]:y_size * index[i + 1],:] = results[i][0]            
            Bi = self.JacobianY_fd(x, y_, y_size *index[i], y_size *index[i+1], self.func_f)       
            Vi = Bi @ results[i][1]
            V = V + Vi
        return y, -V    
    

    def run(
        self,
        x: np.ndarray,
        y: List[np.ndarray],
        max_step: int = 10000,
        step_type: str= 'fixed',
        alpha: float = 1e-3,
        eta: float = 0.8,
        epsilon: float = 1e-8,
        group_size: int = 100,
        num_cores: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main optimization loop.
        Returns:
            x, y: optimized variables
        """
        # Flatten y for parallel processing
        m = y[0].shape[0]
        y_vector = np.concatenate(y, axis=0)
        
        
        if self.is_linear_funcg(self.func_g) and self.is_linear_funcf(self.func_f):
            if self.verbose:
                print("First stage is linear.")
                print("Second stage is linear (linear block structure).")
            group_num = int(np.ceil(len(y) / group_size))
            loop_num = range(0, len(y), group_size)
            index = list(loop_num) + [len(y)]
            lr = np.ones([group_num, 1])
            for i in range(len(index) - 1):
                M_norm = []
                for j in range(index[i], index[i + 1]):
                    eigvals = np.linalg.eigvalsh(self.func_g[j][1].T @ self.func_g[j][1])
                    M_norm.append(np.max(eigvals))
                lr[i, 0] = (group_size*m / max(M_norm)) ** 0.5 if M_norm else 1.0
            y_vector, V = self.stage1_QNSTR_LCP(x, y_vector, index, lr, group_size, m, num_cores)
            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
            res_old = res
            for i in range(max_step):
                if self.verbose:
                    print(f'Iteration {i}, residual: {res:.4e}')
                if res < epsilon:
                    print('Converged!')
                    break
                #lfuncf = [self.func_f[0] + V, self.func_f[1] @ y_vector + self.func_f[2] - V @ x]
                lM = self.func_f[0] + V
                lq = self.func_f[1] @ y_vector + self.constant_f - V @ x
                subproblem2 = QNSTRLCP(lM, lq, self.domain_f)
                x_bar = subproblem2.run(x, epsilon=1e-12, max_step=1000, display=False)
                y_vector_, V = self.stage1_QNSTR_LCP(x_bar, y_vector, index, lr, group_size, m, num_cores)
                res = self.residual(x_bar, y_vector_, self.func_f, self.constant_f, self.domain_f)
                if res > eta * res_old:
                    print('EG (extragradient) correction begin')
                    if step_type == 'fixed':                      
                        for j in range(500):
                            x_ = proj(x - alpha * (self.func_f[0] @ x + self.func_f[1] @ y_vector + self.constant_f), self.domain_f)
                            y_vector, _ = self.stage1_QNSTR_LCP(x_, y_vector, index, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha * (self.func_f[0] @ x_ + self.func_f[1] @ y_vector + self.constant_f), self.domain_f)
                            y_vector, V = self.stage1_QNSTR_LCP(x_bar, y_vector, index, lr, group_size, m, num_cores)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break   
                    elif step_type == 'BB':
                        for j in range(500):
                            func_f_pre = self.func_f[0] @ x + self.func_f[1] @ y_vector
                            x_ = proj(x - alpha * (func_f_pre + self.constant_f), self.domain_f)
                            y_vector, _ = self.stage1_QNSTR_LCP(x_, y_vector, index, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha * (self.func_f[0] @ x_ + self.func_f[1] @ y_vector + self.constant_f), self.domain_f)
                            y_vector, V = self.stage1_QNSTR_LCP(x_bar, y_vector, index, lr, group_size, m, num_cores)
                            BB_sk = x_bar - x
                            BB_gk = self.func_f[0] @ x_bar + self.func_f[1] @ y_vector - func_f_pre
                            alpha = (BB_sk.T@BB_sk)/(BB_sk.T@BB_gk)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break  
                    elif step_type == 'line_search':
                        for j in range(500):
                            alpha_k = 100*alpha
                            self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            y_vector_ = y_vector
                            for _ in range(500):
                                x_ = proj(x - alpha_k * (self.func_f[0] @ x + self.func_f[1] @ y_vector + self.constant_f), self.domain_f)
                                y_vector, _ = self.stage1_QNSTR_LCP(x_, y_vector, index, lr, group_size, m, num_cores)
                                x_bar = proj(x - alpha_k * 
                                             (self.func_f[0] @ x_ + self.func_f[1] @ y_vector + self.constant_f), self.domain_f)
                                y_vector, V = self.stage1_QNSTR_LCP(x_bar, y_vector, index, lr, group_size, m, num_cores)
                                res = self.residual(x_bar, y_vector, self.func_f, self.constant_f, self.domain_f)
                                if res < res_old:
                                    x = x_bar
                                    break
                                else:
                                    alpha_k = alpha_k * 0.9
                                    y_vector = y_vector_
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break
                    else:
                        print("请输入：'fixed'，'BB', 'line_search' ")
                        sys.exit(1)  # 终止程序，返回非零状态码                                     
                else:
                    res_old = res
                    x = x_bar
                    y_vector = y_vector_
                    
        elif self.is_linear_funcg(self.func_g) and self.is_linear_funcf(self.func_f) == False:
            if self.verbose:
                print("First stage is nonlinear.")
                print("Second stage is linear (linear block structure).")
            group_num = int(np.ceil(len(y) / group_size))
            loop_num = range(0, len(y), group_size)
            index = list(loop_num) + [len(y)]
            lr = np.ones([group_num, 1])
            for i in range(len(index) - 1):
                M_norm = []
                for j in range(index[i], index[i + 1]):
                    eigvals = np.linalg.eigvalsh(self.func_g[j][1].T @ self.func_g[j][1])
                    M_norm.append(np.max(eigvals))
                lr[i, 0] = (group_size*m / max(M_norm)) ** 0.5 if M_norm else 1.0
            
            y_vector, V = self.stage1_QNSTR_LCP1(x, y_vector, index, lr, group_size, m, num_cores)
            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
            res_old = res
            for i in range(max_step):
                if self.verbose:
                    print(f'Iteration {i}, residual: {res:.4e}')
                if res < epsilon:
                    print('Converged!')
                    break
                #lfuncf = [self.func_f[0] + V, self.func_f[1] @ y_vector + self.func_f[2] - V @ x]
                lM = self.Jacobian_fd(x, y_vector, self.func_f) + V
                lq = self.func_f(x, y_vector) - self.func_f(x , np.zeros_like(y_vector))+ self.constant_f - V @ x
                subproblem2 = QNSTRLCP(lM, lq, self.domain_f)
                x_bar = subproblem2.run(x, epsilon=1e-12, max_step=1000, display=False)
                y_vector_, V = self.stage1_QNSTR_LCP1(x_bar, y_vector, index, lr, group_size, m, num_cores)
                res = self.residual(x_bar, y_vector_, self.func_f, self.constant_f, self.domain_f)
                if res > eta * res_old:
                    print('EG (extragradient) correction begin')
                    if step_type == 'fixed':                      
                        for j in range(500):
                            x_ = proj(x - alpha * (self.func_f(x, y_vector) + self.constant_f), self.domain_f)
                            y_vector, _ = self.stage1_QNSTR_LCP1(x_, y_vector, index, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                            y_vector, V = self.stage1_QNSTR_LCP1(x_bar, y_vector, index, lr, group_size, m, num_cores)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break
                    elif step_type == 'BB':
                        alpha_k = alpha
                        for j in range(500):
                            func_f_pre = self.func_f(x, y_vector)
                            x_ = proj(x - alpha_k * (func_f_pre + self.constant_f), self.domain_f)
                            y_vector, _ = self.stage1_QNSTR_LCP1(x_, y_vector, index, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha_k * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                            y_vector, V = self.stage1_QNSTR_LCP1(x_bar, y_vector, index, lr, group_size, m, num_cores)
                            BB_sk = x_bar - x
                            BB_gk = self.func_f(x_bar, y_vector) - func_f_pre
                            alpha_k = (BB_sk.T@BB_gk)/(BB_gk.T@BB_gk)
                            #alpha_k = (BB_sk.T@BB_sk)/(BB_sk.T@BB_gk)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break
                    elif step_type == 'line_search':
                        for j in range(500):
                            alpha_k = 100*alpha
                            res_k = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            print('res_k: '+str(res_k))
                            y_vector_ = y_vector
                            for _ in range(500):
                                x_ = proj(x - alpha_k * (self.func_f(x, y_vector) + self.constant_f), self.domain_f)
                                y_vector, _ = self.stage1_QNSTR_LCP1(x_, y_vector, index, lr, group_size, m, num_cores)
                                x_bar = proj(x - alpha_k * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                                y_vector, V = self.stage1_QNSTR_LCP1(x_bar, y_vector, index, lr, group_size, m, num_cores)
                                res = self.residual(x_bar, y_vector, self.func_f, self.constant_f, self.domain_f)
                                if res < res_k:
                                    x = x_bar
                                    break
                                else:
                                    alpha_k = alpha_k * 0.9
                                    y_vector = y_vector_
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break
                    else:
                        print("请输入：'fixed'，'BB', 'line_search' ")
                        sys.exit(1)  # 终止程序，返回非零状态码                        
                else:
                    res_old = res
                    x = x_bar
                    y_vector = y_vector_
                                        
                    
        else:
            if self.is_linear_funcf(self.func_f) == False:
                func_f = lambda x, y: self.func_f(x,y) + self.constant_f
            else:
                func_f = lambda x, y: self.func_f[0]@x + self.func_f[1]@y + self.constant_f
            lr = np.ones([len(y), 1])
            lr_ = 1
            m = y[0].shape[0]
            y_vector, lr = self.stage1_QNSTR(x, y_vector, lr, group_size, m, num_cores) 
            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
            res_old = res     
            for i in range(max_step):
                # TODO: 实现非线性并行y更新  
                if self.verbose:
                    print(f'Iteration {i+1}, residual: {res:.4e}')
                if res < epsilon:
                    print('Converged!')
                    break                  
                subproblem = QNSTR(func_f, self.domain_f)
                x_bar, lr_ = subproblem.run(x, y_vector, epsilon=1e-12, lr=lr_, max_step=500, display=False) 
                y_vector_, lr = self.stage1_QNSTR(x_bar, y_vector, lr, group_size, m, num_cores)
                res = self.residual(x_bar, y_vector_, self.func_f, self.constant_f, self.domain_f)
                if res > eta * res_old:
                    print('EG (extragradient) correction begin')
                    if step_type == 'fixed':                      
                        for j in range(500):
                            x_ = proj(x - alpha * (self.func_f(x, y_vector) + self.constant_f), self.domain_f)
                            y_vector, lr = self.stage1_QNSTR(x_, y_vector, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                            y_vector, lr = self.stage1_QNSTR(x_bar, y_vector, lr, group_size, m, num_cores)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break    
                    elif step_type == 'BB':
                        for j in range(500):
                            func_f_pre = self.func_f(x, y_vector)
                            x_ = proj(x - alpha * (func_f_pre + self.constant_f), self.domain_f)
                            y_vector, lr = self.stage1_QNSTR(x_, y_vector, lr, group_size, m, num_cores)
                            x_bar = proj(x - alpha * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                            y_vector, lr = self.stage1_QNSTR(x_bar, y_vector, lr, group_size, m, num_cores)
                            BB_sk = x_bar - x
                            BB_gk = self.func_f(x_bar, y_vector) - func_f_pre
                            alpha = (BB_sk.T@BB_sk)/(BB_sk.T@BB_gk)
                            x = x_bar
                            res = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break                                
                    elif step_type == 'line_search':
                        for j in range(500):
                            alpha_k = 100*alpha
                            res_k = self.residual(x, y_vector, self.func_f, self.constant_f, self.domain_f)
                            y_vector_ = y_vector                            
                            for _ in range(500):
                                x_ = proj(x - alpha_k * (self.func_f(x, y_vector) + self.constant_f), self.domain_f)
                                y_vector, lr = self.stage1_QNSTR(x_, y_vector, lr, group_size, m, num_cores)
                                x_bar = proj(x - alpha_k * (self.func_f(x_, y_vector) + self.constant_f), self.domain_f)
                                y_vector, lr = self.stage1_QNSTR(x_bar, y_vector, lr, group_size, m, num_cores)
                                res = self.residual(x_bar, y_vector, self.func_f, self.constant_f, self.domain_f)
                                if res < res_old:
                                    x = x_bar
                                    break
                                else:
                                    alpha_k = alpha_k * 0.9
                                    y_vector = y_vector_
                            if self.verbose:
                                print(f'EG step {j}, residual: {res:.4e}')
                            if res < eta * res_old:
                                res_old = res
                                break
                    else:
                        print("请输入：'fixed'，'BB', 'line_search' ")
                        sys.exit(1)  # 终止程序，返回非零状态码                                   
                else:
                    res_old = res
                    x = x_bar
                    y_vector = y_vector_
    
        return x, y_vector

def main():
    np.random.seed(122)
    n = 50
    m = 10
    N = 100
    # 生成测试数据
    A, B, T, M, q, q_, p = get_two_SLCP(N, n, m)
    pB = np.concatenate([p[i, 0] * B[i, 0] for i in range(N)], axis=1)
    domain_f = [np.zeros((n, 1)), 20 * np.ones((n, 1))]
    domain_g = [[np.zeros((m, 1)), 1 * np.ones((m, 1))] for _ in range(N)]
    x_init = np.random.uniform(size=(n, 1))
    x_init = np.clip(x_init, domain_f[0], domain_f[1])
    y_init = [np.zeros((m, 1)) for _ in range(N)]
    G_ = [[T[i, 0], M[i, 0], q_[i, 0]] for i in range(N)]
    def H(x,y):
        return A@x+pB@y

    G = [lambda y,x, i=i: T[i,0]@x + M[i,0]@y + q_[i,0] for i in range(0, N)]
    start_time = time.time()
    solver = PSNASolver(
        func_f = [A, pB],
        constant_f = q,
        func_g = G_,
        #func_f=[A, pB, q],
        #func_g=G_,
        domain_f=domain_f,
        domain_g=domain_g,
        verbose=True
    )
    x_, y_ = solver.run(
        x=x_init,
        y=y_init,
        step_type='fixed',
        max_step=100,
        group_size=20,
        num_cores=-1
    )
    print('Time cost:', time.time() - start_time)
    print('x solution:', x_)

if __name__ == "__main__":
    main()
