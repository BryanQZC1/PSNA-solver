import pandas as pd
import numpy.matlib
from autograd import grad
#import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad
from sklearn import preprocessing 
import copy
import math
from scipy.linalg import block_diag
import time

from joblib import Parallel, delayed
import multiprocessing
import cProfile

from warm_up import QX_solver


class QNSTR:
    """
    Main class for the QNSTR algorithm

    Args:
        loss (callable): Objective function, e.g., f(x, y)
        domain (list): Variable constraint intervals, e.g., [[-2,2], [-2,2]]
    """

    def __init__(self, func, domain):
        self.func = func
        self.domain = domain

    #Define functions
    def get_dim(self, var):
        """
        Get the total number of elements in a tensor
    
        Args:
            var (np.ndarray): numpy array
    
        Returns:
            int: Number of elements
        """
        return np.prod(np.array(var.shape))
    
    def get_dim_list(self, t_vars):
        """
        Get the total number of elements in a list of tensors
    
        Args:
            t_vars (list): List of variables
    
        Returns:
            int: Total number of elements
        """
        return sum([self.get_dim(var) for var in t_vars])
    
    
    
    def smooth_approximation_plus(self, x, kappa2):
        '''smoothing approximation of the function: max(x,0)'''  
        F1 = (x + kappa2/2)**2/(kappa2*2)
        F2 = x
        F3 = 0
        Ft = F1*(abs(x)<=(kappa2/2))+F2*(x>(kappa2/2))+F3*(x<-(kappa2/2))
        #F1 = x**2/(2*kappa2)
        #F2 = x - kappa2/2
        #F3 = 0
        #Ft = F1*(x>=0)*(x<=kappa2)+F2*(x>kappa2)+F3*(x<0)
        return Ft
    
    
    
    def smooth_approximation_mid(self, V, kappa2):
        '''smoothing approximation of the 'mid' function: mid(x,l,u)'''
        domain1 = [copy.deepcopy(self.domain[0]), copy.deepcopy(self.domain[1])]
        domain1[0][self.domain[0]==-np.inf] = 0
        domain1[1][self.domain[1]==np.inf] = 0
        F1 = -self.smooth_approximation_plus(V - domain1[1], kappa2)*(self.domain[1]<np.inf) + self.smooth_approximation_plus(domain1[0] - V, kappa2)*(self.domain[0]>-np.inf) + V
        return F1
    
    
    def F(self, x, kappa2, lr, *args):
        '''smoothing approximation of function F(x):=x-mid(x,l,u)'''
        q = x - lr*self.func(x, *args)
        F = x - self.smooth_approximation_mid(q, kappa2)
        return F
    
    
    
    def F_(self, x, *args):          
        '''smoothing approximation of F(x):=x-mid(x,l,u)'''
        q = x - self.func(x, *args)
        F1 = x- (np.maximum(self.domain[0]-q,0) + q - np.maximum(q-self.domain[1],0))
        return F1
    
    
    
    # r(z,\mu)
    def least_square_smooth(self, x, kappa2, lr, *args):# 输入F对应超参数
        """
        Least squares objective for smoothed F(x), returns 0.5*||F(x)||^2.
    
        Args:
            x (list): List of variables
            mu_s (float): Smoothing parameter
            domain (list): Variable constraint intervals
    
        Returns:
            float: Scalar
        """
        smooth_F = self.F(x, kappa2, lr, *args)# 输入F对应超参数
        result = smooth_F.T@smooth_F
        return 0.5*result 
         
    
    #\nabla r(z,\mu)
    #def grad1(self, x, kappa2, lr, *args):# 输入F对应超参数
    #    """
    #    Compute the gradient of the smoothed least squares objective.
    #
    #    Args:
    #        x (list): List of variables
    #        mu_s (float): Smoothing parameter
    #        domain (list): Variable constraint intervals
    #
    #    Returns:
    #        np.ndarray: Gradient vector
    #    """
    #    grad = elementwise_grad(self.least_square_smooth)
    #    dx = grad(x, kappa2, lr, *args)# 输入F对应超参数
    #    return dx
    
    ######################################################
    def vector_vector(self, x, Fk, kappa2, lr, *args):# 输入F对应超参数
        '''value of a.T*F~(x,kappa2), where a is an arbitrary vector.'''
        #a = vectorwise_(x, bound, kappa2, lr)# 输入F对应超参数
        a = self.F(x, kappa2, lr, *args)# 输入F对应超参数
        return a.T@Fk
    
    
    def grad2(self, x, Fk, kappa2, lr, *args):# 输入F对应超参数
        """
        Compute the gradient of F(x) @ y1 (for quasi-Newton update)
    
        Args:
            x (list): List of variables
            mu_s (float): Smoothing parameter
            domain (list): Variable constraint intervals
            y1 (np.ndarray): Vector
    
        Returns:
            np.ndarray: Gradient
        """
        grad = elementwise_grad(self.vector_vector)
        Jk_v = grad(x, Fk, kappa2, lr, *args)# 输入F对应超参数
        return Jk_v
    
    
    
    def grad_J_J1(self, x, v, Fk, kappa2, epsilon_bar, lr, *args):# 输入F对应超参数
        '''compute the Jk*Vk (compare with grad_J_J, some redundant computations are removed, so it is faster.)'''
        norm_y = np.linalg.norm(v)
        if norm_y!=0:
            epsilon_bar_ = epsilon_bar/norm_y
        else:
            epsilon_bar_ = epsilon_bar    
        #x = update_parameter_epsilon(x, epsilon_bar_, y)
        Fk1 = self.F(x+epsilon_bar_*v, kappa2, lr, *args)# 输入F对应超参数
        Jk_v = (Fk1-Fk)/(epsilon_bar_)
        return Jk_v
    
    
    def vectors_matrix(self, a, b, c):
        """
        Vector-matrix transformation for quasi-Newton formula.
    
        Args:
            a (np.ndarray): Vector
            b (np.ndarray): Vector
            c (np.ndarray): Vector
    
        Returns:
            np.ndarray: Transformation result
        """
        #dim = get_dim(a)
        op = (a.T@c)/(a.T@b) * a
        return op
                

    def GN_BFGS(self, vk, Ak_sk, sk_vector, zk_vector, L):
        """
        Quasi-Newton (BFGS) direction update.
    
        Args:
            vk (np.ndarray): Current vector
            Ak_sk (np.ndarray): Historical information
            sk_vector (np.ndarray): Historical information
            zk_vector (np.ndarray): Historical information
            L (int): Step count
    
        Returns:
            np.ndarray: Update result
        """
        result = vk
        for i in range(L):
            result = result - self.vectors_matrix(Ak_sk[:,i:i+1], sk_vector[:,i:i+1], vk) + self.vectors_matrix(zk_vector[:,i:i+1], sk_vector[:,i:i+1], vk)
        return result    
    
    
    
    
    def GN_BFGS_two_matrix(self, Vk, Ak_sk, sk_vector, zk_vector, e, zk_sk_value, epsilon_criteria, rk_norm):
        '''quasi-Newton matrix update'''
        if zk_sk_value >= epsilon_criteria:
            result = self.GN_BFGS(Vk, Ak_sk, sk_vector, zk_vector, e)
        else:
            result = min(rk_norm, 1e-2) * Vk
        return result 
    
    
    def update_Qk(self, x, Fk, Hk_vector, Vk, L, kappa2, epsilon_bar, lr, *args):# 输入F对应超参数
        """
        Second-order matrix Qk update for trust region subspace.
    
        Args:
            x (list): List of variables
            Hk_vector (np.ndarray): BFGS direction
            Vk (np.ndarray): Historical gradient matrix
            zk_sk_value (float): BFGS criterion
            epsilon_criteria (float): BFGS criterion threshold
            rk_norm (float): Residual norm
            L (int): Subspace dimension
            mu_s (float): Smoothing parameter
            domain (list): Variable constraint intervals
    
        Returns:
            np.ndarray: Qk
        """
        for i in range(L):
            if i==0:
                Jk_vector = self.grad_J_J1(x, Vk[:,i:i+1], Fk, kappa2, epsilon_bar, lr, *args)# 输入F对应超参数
            else:
                Jk_vector = np.concatenate((Jk_vector, self.grad_J_J1(x, Vk[:,i:i+1], Fk, kappa2, epsilon_bar, lr, *args)),1)
    
        Qk = Jk_vector.T@Jk_vector + Vk.T@Hk_vector
        Qk = 0.5*(Qk.T+Qk)
        return Qk
    
    
    
    def update_Vk_d(self, Vk, gk, xk, xk_):
        '''update of searching subspace Vk: with dk = x_{k}-x_{k-1}'''
        dk = xk-xk_
        dim,L=Vk.shape
        Vk[:,0:1] = -gk.reshape([dim,1], order="F")        
        if L>2:
            Vk[:,2:L] = Vk[:,1:L-1]      
        #for i in range(L-1):
        #    Vk[:,L-i:L+1-i] = Vk[:,L-i-1:L-i].reshape([dim,1], order="F")
        Vk[:,1:2] = dk.reshape([dim,1], order="F")
        return Vk
    
    
    def update_Vk_F(self, Vk, gk, Fk):
        '''update of searching subspace Vk: with dk = F~(xk,kappa2)'''
        dim,L=Vk.shape
        Vk[:,0:1] = -gk.reshape([dim,1], order="F")
        if L>2:
            Vk[:,2:L] = Vk[:,1:L-1]
        #for i in range(L-1):
        #    Vk[:,L-i:L+1-i] = Vk[:,L-i-1:L-i].reshape([dim,1], order="F")
        #Vk[:,1:2] = Fk.reshape([dim,1], order="F")
        Vk[:,1:2] = Fk.reshape([dim,1], order="F")
        return Vk    
    
    
    def update_Vk_g(self, Vk, gk):
        '''update of searching subspace Vk: with dk = gk'''
        dim,L=Vk.shape
        Vk[:,1:L] = Vk[:,0:L-1]
        #for i in range(L):
        #    Vk[:,L-i:L-i+1] = Vk[:,L-i-1:L-i].reshape([dim,1], order='F')
        Vk[:,0:1] = -gk.reshape([dim,1], order='F')
        return Vk
    
    
    def update_Vk_Jacobian(self, Vk, gk, xk, Fk, kappa2, epsilon_bar, lr, *args):
        '''update of searching subspace Vk: with dk = Jk*Fk'''
        #dk = vectorwise(zk)-vectorwise(zk_)
        dim,L=Vk.shape
            
        Vk[:,0:1] = -gk#.reshape([dim,1], order="F")
        Vk[:,1:2] = Fk#.reshape([dim,1], order="F")
        F_new = Fk
        #Vk[:,2:3] = H(zk)
        if L>2:
            for i in range(2,L):
                F_new = self.grad_J_J1(xk, F_new, Fk, kappa2, epsilon_bar, lr, *args)
                Vk[:,i:i+1] = F_new#.reshape([dim,1], order="F")
        #Vk[:,L-1:L] = dk.reshape([dim,1], order="F")
        return Vk    
    
     
        
    def update_Vk_Jacobian_T(self, Vk, gk, xk, Fk, kappa2, epsilon_bar, lr, *args):
        '''update of searching subspace Vk: with dk = Jk.T*Fk'''
        #dk = vectorwise(zk)-vectorwise(zk_)
        dim,L=Vk.shape
            
        Vk[:,0:1] = -gk.reshape([dim,1], order="F")
        Vk[:,1:2] = - Fk.reshape([dim,1], order="F")
        #Vk[:,2:3] = H(zk)
        if L>2:
            for i in range(2,L):
                #Fk = grad_J_J1(xk, Fk, bound, kappa2, epsilon_bar, lr, func, *args)
                Fk =  self.grad2(xk, Fk, kappa2, lr, *args)
                Vk[:,i:i+1] = Fk.reshape([dim,1], order="F")
        #Vk[:,L-1:L] = dk.reshape([dim,1], order="F")
        return Vk        
            
     
    def update_Vk_Hessian_approx(self, Vk, gk, xk, kappa2, epsilon_bar, lr, *args):
        '''update of searching subspace Vk: with dk = (Jk.T*Jk)*gk'''
        #dk = vectorwise(zk)-vectorwise(zk_)
        dim,L=Vk.shape
            
        Vk[:,0:1] = -gk.reshape([dim,1], order="F")
        #Vk[:,1:2] = - Fk.reshape([dim,1], order="F")
        #Vk[:,2:3] = H(zk)
        Fk = -gk
        if L>1:
            for i in range(1,L):
                Fk = self.grad_J_J1(xk, Fk, kappa2, epsilon_bar, lr, *args)
                Fk =  self.grad2(xk, Fk, kappa2, lr, *args)
                Vk[:,i:i+1] = Fk.reshape([dim,1], order="F")
        #Vk[:,L-1:L] = dk.reshape([dim,1], order="F")
        return Vk 
    
        
    
    def gram_schmidt(self, Vk):
        '''gram schmidt operation to make the column vectors in Vk be orthogonal'''
        for i in range(Vk.shape[1]):
            Vk_gs = 0.
            for j in range(i):
                Vk_gs = Vk_gs + Vk[:,j:j+1].T@Vk[:,i:i+1]/(np.linalg.norm(Vk[:,j:j+1])+1e-16)*Vk[:,j:j+1]
            Vk[:,i:i+1] = Vk[:,i:i+1]-Vk_gs
        return Vk
    
       
    def stable_gram_schmidt(self, Vk):
        '''gram schmidt operation to make the column vectors in Vk be orthogonal'''
        for i in range(Vk.shape[1]):
            Vk_gs = 0.
            for j in range(i):
                Vk_gs = Vk_gs + Vk[:,j:j+1].T@Vk[:,i:i+1]/(np.linalg.norm(Vk[:,j:j+1])+1e-32)*Vk[:,j:j+1]
            Vk[:,i:i+1] = Vk[:,i:i+1]/(np.linalg.norm(Vk[:,i:i+1])+1e-32)-Vk_gs
        return Vk
    
                    
    def normalize_Vk(self, Vk):
        '''Normalization for the Vk to let each column vectors in Vk satisfied: ||Vk(:,i)||=1 for i=1,...L1'''
        dim,L=Vk.shape
        #Vk_ = np.zeros([dim,L])
        for i in range(L):
            Vk_norm = np.linalg.norm(Vk[:,i:i+1])
            Vk[:,i:i+1] = Vk[:,i:i+1]/(Vk_norm+1e-32)
        return Vk
    
    
    def update_Gk(self, Vk):
        """
        Compute Gk = Vk^T Vk
    
        Args:
            Vk (np.ndarray): Historical gradient matrix
    
        Returns:
            np.ndarray: Gk
        """
        Gk = Vk.T@Vk
        Gk = 0.5*(Gk.T+Gk)
        return Gk
    
    def update_ck(self, Vk, gk):
        """
        Compute ck = Vk^T gk
    
        Args:
            Vk (np.ndarray): Historical gradient matrix
            gk (np.ndarray): Current gradient
    
        Returns:
            np.ndarray: ck
        """
        ck=Vk.T@gk
        return ck
    
    
    def update_vk(self, x, xk, Fk, kappa2, lr, rk_norm, rk_norm_, *args):# 输入F对应超参数
        """
        Update for quasi-Newton direction
    
        Args:
            x (list): Current variables
            xk (list): Historical variables
            Fk (np.ndarray): F(x)
            rk_norm (float): Current residual norm
            rk_norm_ (float): Historical residual norm
            mu_s (float): Smoothing parameter
    
        Returns:
            np.ndarray: vk
        """
        Jk_gk = self.grad2(x, Fk, kappa2, lr, *args)# 输入F对应超参数
        Jk1_gk = self.grad2(xk, Fk, kappa2, lr, *args)# 输入F对应超参数
        
        return (Jk_gk - Jk1_gk)*(rk_norm/rk_norm_), Jk_gk
    
    

    
    
    def update_vector(self, x, v, L):
        """
        Insert historical vector
    
        Args:
            x (np.ndarray): Historical vector matrix
            v (np.ndarray): New vector
            L (int): Insert position
    
        Returns:
            np.ndarray: Updated matrix
        """
        #dim = get_dim(v)
        x[:,L:L+1] = v#.reshape([dim,1], order="F")
        return x
    
    def update_matrix(self, x, v):
        '''update value of matrix'''
        dim, L = x.shape
        x[:,0:L-1] = x[:,1:L]
        x[:,L-1:L] = v#.reshape([dim,1])
        return x
    
    
    def Lambda_k(self, Qk, Gk, gamma):
        '''update the Lagrangian multiplier lambda_k'''
        dim=Qk.shape[0]
        eigh,_ = np.linalg.eigh(Qk)
        lmin = min(eigh)
        lmax = max(eigh)
        eigh_,_= np.linalg.eigh(Gk)
        minG = min(eigh_)
        lb = max(0,-lmin/minG)
        ub = max(lb,lmax) + 1e4
        lambda_k = gamma*ub + (1-gamma)*lb
        return lambda_k
    
    
    def inverse(self, Qk, ck):
        """
        Solve the linear system Qk x = ck
    
        Args:
            Qk (np.ndarray):
            ck (np.ndarray):
    
        Returns:
            np.ndarray: Solution
        """
        L = np.linalg.cholesky(Qk)
        y= np.linalg.solve(L, ck)
        zin_kj = np.linalg.solve(L.T, y)    
        return zin_kj
    
    def QNSTR_subproblem(self, Qk,Gk,ck, gamma):
        """
        Trust region subproblem solution in multi-dimensional subspace.
    
        Args:
            Qk (np.ndarray):
            Gk (np.ndarray):
            ck (np.ndarray):
            Delta (float):
            L (int):
            L1 (int):
            epsilon (float):
    
        Returns:
            np.ndarray: alpha_k
        """
        lambda_k = self.Lambda_k(Qk, Gk, gamma)
        #alpha_k = -np.linalg.solve(Qk+lambda_k*Gk, ck)
        try:
            alpha_k = -self.inverse(Qk+lambda_k*Gk, ck)
        except:
            print('Qk: '+str(Qk))
            print('Gk: '+str(Gk))
            print('lambda: '+str(lambda_k))
            alpha_k = -np.linalg.solve(Qk+lambda_k*Gk, ck)
        #if alpha_k.shape[0]<L1:
        #    a = L1-alpha_k.shape[0]
        #    alpha_k = np.vstack((alpha_k, np.zeros([a,1])))
        return alpha_k, lambda_k
    
          
    
    def update_parameter(self, x, alpha, Vk):
        """
        Update variable x with step size alpha and direction Vk.
    
        Args:
            x (list): List of variables
            alpha (np.ndarray): Step size
            Vk (np.ndarray): Direction
    
        Returns:
            list: Updated x
        """
        pk = Vk@alpha
        x = x + pk
        '''
        for i in range(len(x)):
            dim =  get_dim(x[i])
            pk_ = pk[dim1:dim1+dim,0].reshape(x[i].shape, order="F")
            x[i] = x[i] + pk_
            dim1 = dim1+dim'''
        
        return x        
    
    
    def update_parameter_epsilon(self, x, epsilon, Vk):
        pk = epsilon*Vk
        x = x + pk
        #dim1 = 0
        #for i in range(len(x)):
        #    dim =  get_dim(x[i])
        #    pk_ = pk[dim1:dim1+dim,0].reshape(x[i].shape, order="F")
        #    x[i] = x[i] + pk_
        #    dim1 = dim1+dim
        
        return x   
            
    
    def QNSTR_subproblem_1d(self, Qk, ck, ls):
        ''' 1 dimensional subproblem'''
        dim = Qk.shape[0]
        Q11 = Qk[0,0]
        if Q11>0:
            alpha = -ck[0,0]/Q11 * ls
            #alpha = ls
            #dq_value = -alpha*Q11*alpha - alpha*ck[0,0]
        else:
            alpha = 0.1*ls
            #dq_value = alpha* ck[0,0]
        alpha = np.vstack((alpha, np.zeros((dim-1,1))))
        return alpha



    def run(
        self,
        x,     
        *args,
        kappa2 = 1e-2, 
        zeta1 = 0.1, 
        zeta2 = 0.3, 
        beta1 = 0.2, 
        beta2 = 2,
        eta = 0.05, 
        nu = 1, 
        tau = 0.9, 
        epsilon = 1e-8, 
        epsilon_criteria = 1e-6, 
        L = 20, 
        L1= 5, 
        lr= 1,
        max_step = 10000,
        Gamma_ini = 1e-32,
        mode = 'Jacobian',
        normalized = True,
        gram_schmidt = True,
        epsilon_bar = 1e-8,
        warm_up = False,
        display = False):# 输入F对应超参数
        
        """
        Main QNSTR process: Trust region quasi-Newton method for solving saddle point/VI problems

        Args:
            x (list): Initial point (list of variables)
            zeta1, zeta2, beta1, beta2, eta, nu, tau, epsilon, epsilon_criteria (float): Hyperparameters
            memory_size (int): Subspace dimension (original L)
            bfgs_dir_count (int): Number of BFGS directions (original L1)
            max_step (int): Maximum number of iterations
            mu_s (float): Smoothing parameter

        Returns:
            tuple: (final point x, optimization trajectory)
        """        

        if warm_up:
            x_init = x
            x = QX_solver(func = self.func, domain= self.domain,).run(
                x_init,
                *args
            )


        if mode == 'd':
            L1 = 1
        
        dim = self.get_dim(x) 
    
        rk_origin = np.linalg.norm(self.F_(x, *args))
        i = 0
        ii = 0
        ls = 1.
        zk_sk_value = 0.
        Vk = np.zeros([dim, L1])
        sk_vector = np.ones([dim, L])
        Ak_sk = np.ones([dim, L])
        zk_vector = np.ones([dim,L])
        while i < max_step:
            #print('restart!')
            #gk = self.grad1(x, kappa2, lr, *args)# 输入F对应超参数
            Fk = self.F(x, kappa2, lr, *args)
            gk = self.grad2(x, Fk, kappa2, lr, *args)
            gk_norm = np.linalg.norm(gk)
            if rk_origin<epsilon or gk_norm<=1e-2*epsilon:
                print('step cost: '+str(i))
                print('r(x): '+str(rk_origin))
                print('||gk||: ' +str(gk_norm))
                print('converged!!!')
                break         
            e = 0
    
            zk_sk_value = 0
            gamma = Gamma_ini
            #lambda_k = 0.
            while e<=L-1:    
                #print('This is the ' +str(i) +' th step') ###########
                rk_origin = np.linalg.norm(self.F_(x, *args))# 输入F对应超参数
                #rk_value = self.least_square_smooth(x, kappa2, lr, *args)
                Fk = self.F(x, kappa2, lr, *args)
                gk = self.grad2(x, Fk, kappa2, lr, *args)
                gk_norm = np.linalg.norm(gk)
                #print('||gk||: '+str(gk_norm))############
                #print('------------------------')
                if rk_origin < epsilon or gk_norm < 1e-2*epsilon or i >= max_step:
                    if i>=100:
                        lr = lr*0.5
                        print('function is too slow for this problem')
                        print('smoothing parameter updated; kappa2: '+str(kappa2))
                    #else:
                        #print('step cost: '+str(i))
                        #print('r(x): '+str(rk_origin))
                        #print('converged!!!')
                        #print('smoothing parameter updated; kappa2: '+str(kappa2))
                    break      
                    
                if mode == "d":    
                    Vk = self.update_Vk_d(Vk, gk, x, xk)   
                elif mode == "Jacobian":
                    Vk = self.update_Vk_Jacobian(Vk, gk, x, Fk, kappa2, epsilon_bar, lr,*args)
                elif mode == "Jacobian-T":
                    Vk = self.update_Vk_Jacobian_T(Vk, gk, x, Fk, kappa2, epsilon_bar, lr, *args)     
                elif mode == "Hessian-approx":                   
                    Vk = self.update_Vk_Hessian_approx(Vk, gk, x, kappa2, epsilon_bar, lr, *args)                     
                if gram_schmidt==True:
                    Vk = self.stable_gram_schmidt(Vk)               
                if normalized==True:
                    Vk = self.normalize_Vk(Vk)             
                #if np.linalg.norm(x-xk)!=0:    
                
                rk_value_ = np.linalg.norm(Fk)
                Hk_vector = self.GN_BFGS_two_matrix(Vk, Ak_sk, sk_vector, zk_vector, e, zk_sk_value, epsilon_criteria, rk_value_)

                if display:
                    print('rk origin: '+str(rk_origin))
                    print('rk_value: '+str((rk_value_))) ###################
                    print('||gk||: '+str(gk_norm))############
                #print('searching dimension: '+str(L1)) ###################
                Gk = self.update_Gk(Vk)
                ck = self.update_ck(Vk, gk)
                Qk = self.update_Qk(x, Fk, Hk_vector, Vk, L1, kappa2, epsilon_bar, lr, *args)# 输入F对应超参数  
    
                xk = copy.deepcopy(x)
                k = L1
                if k==1:
                    #print('dim = 1')      ################         
                    for ii in range(500):
                        alpha = self.QNSTR_subproblem_1d(Qk,ck, ls)
                        Fk = self.F(xk+Vk@alpha, kappa2, lr, *args)
                        rk_value = np.linalg.norm(Fk)# 输入F对应超参数
                        mka = -(alpha.T@ck + 0.5*alpha.T@Qk@alpha)
                        #rho_k = 0.5*(rk_value_**2 - rk_value**2)/mka

                        #if rho_k>eta and rk_value_ - rk_value>0:
                        if rk_value_**2 - rk_value**2 > 2*eta*mka:
                            x = xk + Vk@alpha
                            ls = 1.
                            gamma = Gamma_ini
                            break
                        else:
                            #x = deal(xk)
                            #x = copy.deepcopy(xk)
                            ls= ls*0.99      
    
                else:
                    #print('dim = ' +str(k))   #################
                    eigh,_ = np.linalg.eigh(Gk)
                    eigh1,_= np.linalg.eigh(Qk)
                    if (min(eigh)>0 and max(eigh)/min(eigh)<=1e8) and min(eigh1)>0:
                        for iii in range(500):
                            alpha, lambda_k = self.QNSTR_subproblem(Qk,Gk,ck, gamma)
                            Fk = self.F(xk+Vk@alpha, kappa2, lr, *args)
                            rk_value = np.linalg.norm(Fk)# 输入F对应超参数   
                            mka = -(alpha.T@ck + 0.5*alpha.T@Qk@alpha)

                            if rk_value_**2 - rk_value**2 < 2*zeta1*mka: 
                                gamma = beta2* gamma
                            #elif rho_k>zeta2:
                            elif rk_value_**2 - rk_value**2 > 2*zeta2*mka:
                                #gamma = max(1e-20,beta1 * gamma)
                                gamma = max(min((gamma)**(1/2), np.log10(gamma + 1)), 1e-20)
                            else:
                                gamma = gamma       
         
    
    
                            #if rho_k>eta and rk_value_ - rk_value>0:
                            if rk_value_**2 - rk_value**2 > 2*eta*mka:
                                x = xk + Vk@alpha               
                                break
    
                dk = x-xk
                norm_dk = np.linalg.norm(dk)**2
                if norm_dk == 0:
                    L1 = max(L1-1, 1)
                    Vk = Vk[:,0:L1]
                    break
                else:
                    #Fk = self.F(x, kappa2, lr, *args)# 输入F对应超参数
                    vk, gk = self.update_vk(x, xk, Fk, kappa2, lr, rk_value, rk_value_, *args)# 输入F对应超参数
                    zk_sk_value = vk.T@dk/norm_dk   
                    
                    if zk_sk_value >= epsilon_criteria:
                        #print('class 1')  ###########
                        BFGS_sk = self.GN_BFGS(dk, Ak_sk, sk_vector, zk_vector, e)
                        sk_vector = self.update_vector(sk_vector, dk, e)
                        Ak_sk = self.update_vector(Ak_sk, BFGS_sk, e)
                        zk_vector = self.update_vector(zk_vector, vk, e)  
                        e = e + 1   
    
                    if 0.8*rk_value_ < rk_value and L1<10:
                        Vk = np.concatenate([Vk,Fk],1)
                        L1 = L1 + 1
                    i=i+1

                    #gk = self.grad2(x, Fk, kappa2, lr, *args)
                    if np.linalg.norm(gk)<=nu*kappa2:
                        kappa2=tau*kappa2
                        #print('smoothing parameter updated; kappa2: '+str(kappa2))
    
        return x, lr
    

   


if __name__ == "__main__":
    
    n = 100   
    '''
    A = np.random.normal(loc=0, scale=1, size=(n,n))
    A = A.T@A #+ 1*np.eye(n)
    #A = A.T@A+ np.eye(n)
    x_star = np.random.normal(loc=0, scale=1, size=(n,1))
    x_star[x_star<0] = 0
    
    b = -A@x_star
    b_eps = np.random.uniform(size=(n,1))
    
    b[x_star==0] = b[x_star==0] + b_eps[x_star==0]
    b[x_star>0] = b[x_star>0]
    
    W = A
    w0= b 
    bound = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]'''
    
    c_0 = np.random.normal(1)# 生成 c_0
    d_0 = np.random.normal(1) # 生成 d_0
    
    Q = np.random.normal(size=(n,n)) 
    Q = Q.T@Q + np.eye(n)
    c = np.random.uniform(size=(n,1))
    d = np.random.uniform(size=(n,1))
    e = np.ones([n,1])
    a = e + c
    b = e + d
    a_0 = 1 + c_0
    b_0 = 1 + d_0

    T = np.random.normal(size=(n,n))
    
    def H(x, y):
        H = ((b.T@x + b_0) * (2*Q@x + a) - b*(x.T@Q@x + a.T@x + a_0))/((b.T@x + b_0)**2) + T@y
        return H
    
    bound=[2.*np.ones([n,1]), 10*np.ones([n,1])]
    '''
    a = np.random.uniform(1)
    #a = np.random.normal(1)
    x_sol = np.random.normal(size=(n,1))
    Q = np.random.normal(size=(n,n))
    x_sol = np.maximum(x_sol,0)
    Q = Q.T@Q + np.eye(n)
    P = np.random.normal(size=(n,n)) 
    P = P.T@P
    q = (-P@x_sol)*(x_sol>0)+(-P@x_sol+np.random.uniform(size=(n,1)))*(x_sol==0)

    bound = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]
    def H(x):
        H = (np.exp(-x.T@Q@x)+a)*(P@x+q)
        return H'''
    
    x_init = np.random.uniform(size=(n,1))
    x_init = np.clip(x_init, bound[0],bound[1])
    y_init = np.random.uniform(size=(n,1))

    start_time = time.time()
    demo = QNSTR(
        #func=lambda x: W@x+w0,
        func = H,
        domain=bound,
    )
    x_ = demo.run(
        x_init,
        y_init,
        lr = 0.6,
        mode = 'Jacobian',
        warm_up = False
    )
    print('time cost: '+str(time.time()-start_time))
    print(x_)