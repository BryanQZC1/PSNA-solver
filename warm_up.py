import pandas as pd
import numpy.matlib
from autograd import grad
import numpy as np
from sklearn import preprocessing 
import copy
import math
from scipy.linalg import block_diag
import time

from joblib import Parallel, delayed
import multiprocessing
import cProfile



class QX_solver:
    """
    Main class for the warm up algorithm, the warm up algorithm is used the algorithm wirtten by QX

    Args:
        loss (callable): Objective function, e.g., f(x, y)
        domain (list): Variable constraint intervals, e.g., [[-2,2], [-2,2]]
    """

    def __init__(self, func, domain):
        self.func = func
        self.domain = domain

    def deal(self, *args):
        A1 = []
        for i in range(len(args)):
            a=copy.deepcopy(args[i])
            A1.append(a)
            #A1.append(args[i])
        return A1[0]


    def G(self, x, domain, H, lr):
        '''smoothing approximation of function F(x):=x-mid(x,l,u)'''
        q = x - lr*H
        G_ = (np.maximum(domain[0]-q,0) + q - np.maximum(q-domain[1],0))
        return G_
    
    

    def run(self,
               x_init,
               *args,
               QX_gamma = 0.6,
               QX_rho = 0.8,
               QX_tau = 0.6,
               QX_M1 = 5000,
               QX_omega = 30,
               QX_sigma0 = 1,
               QX_mu = 0.5, 
               max_iter = 100): 
        
        x_k = self.deal(x_init)
        sigma_k = QX_sigma0
        for k in range(max_iter):
            t1 = QX_gamma
            Gf_x = self.func(x_k, *args)
            residual= np.linalg.norm(self.G(x_k, self.domain, Gf_x, 1) - x_k)
            if residual<=1e-10:
                #print('converge')
                print('Current residula value: '+str(residual))
                print('step cost: '+str(k))
                break
            else:
                
                for m in range(1000):
                    y_k_05 = self.G(x_k, self.domain, Gf_x, t1)
                    Gf_y_05 = self.func(y_k_05, *args)
                    y_k_1 = self.G(x_k, self.domain, Gf_y_05, t1)
                    y_k_05_1 = y_k_05 - y_k_1
                    F_k = y_k_05 - x_k
    
                    if t1 * ((Gf_y_05-Gf_x).T@y_k_05_1)<=(QX_mu/2)*(F_k.T@F_k+y_k_05_1.T@y_k_05_1):
                        break
                    else: 
                        t1 = t1*QX_rho   
                        
                        
                F_tilde_k = y_k_1 - x_k
                nor_F_tilde_k = np.linalg.norm(F_tilde_k)
                
                diff_F_k = F_k - y_k_05_1
                if nor_F_tilde_k < np.linalg.norm(F_k) and nor_F_tilde_k < QX_omega* (sigma_k)**(-1*QX_tau):
                    alpha_k = -(diff_F_k.T@F_tilde_k) / np.linalg.norm(diff_F_k)**2
                else:
                    alpha_k = QX_M1 + 1
                    
                if abs(alpha_k) <= QX_M1:
                    beta_k = 1 - alpha_k
                    x_k = alpha_k * x_k + beta_k * y_k_1
                    sigma_k = sigma_k + 1
                else:
                    x_k = y_k_1
        return x_k




if __name__ == "__main__":
    
    n = 10
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
    '''
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
    
    def H(x):
        H = ((b.T@x + b_0) * (2*Q@x + a) - b*(x.T@Q@x + a.T@x + a_0))/((b.T@x + b_0)**2)
        return H
    
    bound=[2.*np.ones([n,1]), 10*np.ones([n,1])]'''
    
    a = 0.01
    #a = np.random.uniform(1)
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
        return H
    
    x_init = np.random.uniform(size=(n,1))
    x_init = np.clip(x_init, bound[0],bound[1])
    demo = QX_solver(
        #func=lambda x: W@x+w0,
        func = H,
        domain=bound,
    )
    x_ = demo.run(
        x_init,
        max_iter = 10000
    )
    #print(x_, rk_list_)
    print(x_)