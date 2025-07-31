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

from QNSTR import *
from generate_two_stage_LCP import *

def sub_residual(x, domainf, func, *args):
    F = func(x, *args)
    res = np.linalg.norm(x-np.clip(x-F, domainf[0], domainf[1]))**2 
    return res

def proj(x, domain):          
    '''projection onto a box constraint'''
    F1 = (np.maximum(domain[0]-x,0) + x - np.maximum(x-domain[1],0))
    return F1



class two_stage_NVI:
    """
    Main class for the QNSTR algorithm

    Args:
        loss (callable): Objective function, e.g., f(x, y)
        domain (list): Variable constraint intervals, e.g., [[-2,2], [-2,2]]
    """

    def __init__(self, funcf, funcg, domainf, domaing):
        self.funcf = funcf
        self.funcg = funcg
        self.domainf = domainf
        self.domaing = domaing

    
    def residual(self, x, y):
        res = sub_residual(x, self.domainf, self.funcf, y)
        for i in range(len(self.funcg)):
            res = res + sub_residual(y[i], self.domaing[i], self.funcg[i], x)
        return np.sqrt(res)
        
    def update_parameter_parallel(self, y, x, domain, lr, i, func):
        print('this is the ' +str(i+1) + 'th subproblem')
        #group_size=100
        #y_ = QNSTR(y, domain, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-12, epsilon_criteria, L, L1, lr, max_step, Delta_ini, mode, normalized, gram_schmidt, epsilon_bar, func, x)            
        subproblem = QNSTR(func, domain)
        y_ = subproblem.run(y, x, epsilon = 1e-12, lr = lr, max_step = 1000, display = False)  
        #y_ = subproblem.run(y, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-12, epsilon_criteria, L, L1, lr, max_step, Gamma_ini, 
        #                        mode, normalized, gram_schmidt, epsilon_bar, warm_up, x)  
        return y_ 

    #def update_parameter_parallel1(self, y, x, domain, lr, i, func):
    #    print('this is the ' +str(i+1) + 'th subproblem')
    #    #group_size=100
    #
    #    for j in range(len(y)):        
    #        subproblem = QNSTR(func[j], domain[j])
    #        y[j], lr[j,0] = subproblem.run(y[j], x, epsilon = 1e-12, lr = lr[j,0], max_step = 1000, display = False)  
    #    return [y, lr]
    
    '''
    def semi_smooth_hessian(Ni, Di, Mi):
        ind = (abs(y)-epsilon>=0)
        row1 = np.arange(0, n, 1)
        col1 = np.arange(0, n, 1)
        ones = np.ones(n)
        Ix = scipy.sparse_coo_matrix((ones, (row1, col1)), shape=(n,n))
        Di = scipy.sparse_coo_matrix((ind, (row1, col1)), shape=(n,n))
        Vi = Bi@(np.linalg.inv(Ix-Di+Di@Mi)@Di@Ni)'''
    
    
    def stage1_QNSTR(self, x, y, lr_list, group_size, num_cores):
        # Stage 1
        print('stage 2')
        ##################################################
        loop_num = range(0, len(y), 1)
        results = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel)(y[i], x, self.domaing[i], lr_list[i,0], i, self.funcg[i]) for i in loop_num)
    
        for i in loop_num:
            y[i] = results[i][0]    
            lr_list[i,0] = results[i][1]
        ############################################################
        #loop_num = range(0, len(y), int(len(y)/group_size))
        #results, results1 = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel1)(y[i:i+group_size], x, self.domaing[i:i+group_size], lr_list[i:i+group_size,0], i, self.funcg[i:i+group_size]) for i in loop_num)

        #for i in loop_num:
        #    y[i:i+group_size] = results[i:i+group_size]    
        #    lr_list[i:i+group_size,0] = results1[i:i+group_size]       
        '''
        group_size=100
        for i in range(int(T.shape[0]/group_size)):
            for j in range(group_size):
                dim = M[i*group_size+j,0].shape[0]
                if j == 0:
                    y_ = y[i*group_size+j]
                    A1 = M[i*group_size+j,0] + dim/k*np.eye(dim)
                    b1 = T[i*group_size+j,0]@x + q_[i*group_size+j,0]
                    bound_y_ = bound_y[0]
                else:
                    y_ = np.concatenate((y_, y[i*group_size+j]),0)
                    A1 = block_diag(A1, M[i*group_size+j,0] + dim/k*np.eye(dim))
                    b1 = np.concatenate((b1, T[i*group_size+j,0]@x + q_[i*group_size+j,0]),0)
    
            y_ = QNSTR_radius_free(y_, bound_y_, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-8, epsilon_criteria, L, L1, max_step, Delta_ini, mode, normalized, gram_schmidt, epsilon_bar, A1, b1)
            dim=0
            for j in range(group_size):
                dim_y = y[i*group_size+j].shape[0]
                y[i*group_size+j] = y_[dim:dim+dim_y,:]
                dim = dim+dim_y'''

        # Stage 2
        #print('stage 2')
        #subproblem2 = QNSTR(funcf, domainf)
        #x, lr_ = subproblem2.run(x, y, lr = lr_, epsilon = 1e-12, max_step = 1000, display= True)
        return y, lr_list
    
    
    def run(self,
            x, 
            y,  
            max_step = 10000,
            epsilon = 1e-8,
            group_size=1, 
            num_cores=-1):

        lr = 1*np.ones([len(y),1])
        lr_= 0.5
        #res = self.residual(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
        #print('residual: ' +str(res))
        for i in range(max_step):
            y, lr = self.stage1_QNSTR(x, y, lr, group_size, num_cores)
            res = self.residual(x, y)
            '''
            y_ = vectorwise(y)
            ind = np.where(abs(y_)-epsilon>=0)
            row1 = np.arange(0, n, 1)
            col1 = np.arange(0, n, 1)
            ones = np.ones(n)
            Ix = scipy.sparse_coo_matrix((ones, (row1, col1)), shape=(n,n))
            Di = scipy.sparse_coo_matrix((ind, (row1, col1)), shape=(n,n))'''
  
            
            print('residual: ' +str(res))
            if res < epsilon:
                print('converge!!!')
                print('Step cost: '+str(i))                
                break

            print('stage 1')
            subproblem2 = QNSTR(self.funcf, self.domainf)
            x, lr_ = subproblem2.run(x, y, lr = lr_, epsilon = 1e-12, max_step = 1000, display= False)         
                
            #t = 1e-3
            #x_ = proj(x - t*self.funcf(x,y), self.domainf)
            #x_bar = proj(x - t*self.funcf(x_,y), self.domainf)
            #x = x_bar
            
            #res = self.residual(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
            #print('residual: ' +str(res))
            #if res<=epsilon:
            #    print('Converged !!!')
            #    break
    
        return x, y

    

    

   


if __name__ == "__main__":
    n = 100
    m = 10
    N = 1000

    #A, B, T, M, q, q_, p = get_two_SLCP(N, n, m)
    A, B, p, q, Q, T, a, a_0, b, b_0 = get_demo1(N, n, m)
    #A, B, p, q, Q, P, T, q_, a = get_demo2(N, n, m)

    
    domainf = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]
    domaing = []
    for i in range(N):
        domaing.append([2.*np.ones([m,1]), 10*np.ones([m,1])])
        
    x_init = np.random.uniform(size=(n,1))
    x_init = np.clip(x_init, domainf[0],domainf[1])

    y_init = []
    for i in range(N):
        y_init.append(np.zeros([m,1]))

    def H(x,y):
        H1 = A@x + q
        for i in range(len(y)):
            H1 = H1 + p[i,0]*B[i,0]@y[i]
        return H1


    #G = [lambda y,x, i=i: T[i,0]@x + M[i,0]@y + q_[i,0] for i in range(0, N)]
    G = [lambda y,x, i=i: T[i,0]@x + 
         ((b[i,0].T@y + b_0[i,0]) * (2*Q[i,0]@y + a[i,0]) - b[i,0]*(y.T@Q[i,0]@y + a[i,0].T@y + a_0[i,0]))
         /((b[i,0].T@y + b_0[i,0])**2) for i in range(0, N)]
    #G = [lambda y,x, i=i: T[i,0]@x + (np.exp(-y.T@Q[i,0]@y)+a[i,0])*(P[i,0]@y+q_[i,0]) for i in range(0, N)]
    start_time = time.time()    
    demo = two_stage_NVI(
        funcf=H,
        funcg=G,
        domainf=domainf,
        domaing=domaing,
    )
    x_, y_ = demo.run(
        x=x_init,
        y=y_init,
        num_cores = -1
    )
    print('time cost: ' +str(time.time()-start_time))
    print(x_)
