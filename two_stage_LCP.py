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
from QNSTR_LCP1 import *
from generate_two_stage_LCP import *

#def sub_residual(x, domainf, func, *args):
#    F = func(x, *args)
#    res = np.linalg.norm(x-np.clip(x-F, domainf[0], domainf[1]))**2 
#    return res


def proj(x, domain):          
    '''projection onto a box constraint'''
    F1 = (np.maximum(domain[0]-x,0) + x - np.maximum(x-domain[1],0))
    return F1



class two_stage_LCP:
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

        
    def judge(self, funcg):
        result = True
        for i in range(len(funcg)):
            if isinstance(funcg[i], list) == False:
                result = False
                break
        return result
    

    #def sub_residual_LCP(self, x, y, domain, func):
    #    F = func[0]@x + func[1]@y + func[2]
    #    res = np.linalg.norm(y-np.clip(y-F, domain[0], domain[1]))**2 
    #    return res
               
        
    #def residual(self, x, y, domainf, domaing, funcf, funcg):
    #    res = sub_residual(x, domainf, funcf, y)
    #    for i in range(len(self.funcg)):
    #        res = res + sub_residual(y[i], domaing[i], funcg[i], x)
    #    return np.sqrt(res)

    def residual(self, x, y, domainf, func):
        F = func(x, y)
        res = np.linalg.norm(x-np.clip(x-F, domainf[0], domainf[1])) 
        return res    
        
                 
    #def residual_LCP(self, x, y, domainf, domaing, funcf, funcg):
    #    res = sub_residual(x, domainf, funcf, y)
    #    for i in range(len(self.funcg)):
    #        res = res + self.sub_residual_LCP(x, y[i], domaing[i], funcg[i])
    #    return np.sqrt(res)
        
        
    def update_parameter_parallel(self, y, x, domain, lr, i, func):
        print('this is the ' +str(i+1) + 'th subproblem')
        #group_size=100
        #y_ = QNSTR(y, domain, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-12, epsilon_criteria, L, L1, lr, max_step, Delta_ini, mode, normalized, gram_schmidt, epsilon_bar, func, x)            
        subproblem = QNSTR(func, domain)
        y_ = subproblem.run(y, x, epsilon = 1e-12, lr = lr, max_step = 1000, display = False)  
        #y_ = subproblem.run(y, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-12, epsilon_criteria, L, L1, lr, max_step, Gamma_ini, 
        #                        mode, normalized, gram_schmidt, epsilon_bar, warm_up, x)  
        return y_ 


    def update_parameter_parallel_linear_block(self, y, x, domain, lr, s, func):
        print('This is the '+str(s[0]) + '~' +str(s[1]) + 'problems')
        N = len(func)
        for i in range(N):
            if i ==0:
                M = func[i][1]
                q = func[i][0]@x + func[i][2]
                ldomainl = domain[i][0]
                ldomainu = domain[i][1] 
            else:
                M = block_diag(M, func[i][1])
                q = np.concatenate([q, func[i][0]@x + func[i][2]],0)
                ldomainl = np.concatenate([ldomainl, domain[i][0]],0)
                ldomainu = np.concatenate([ldomainu, domain[i][1]],0)
        y_ = y
        ldomain = [ldomainl, ldomainu]
        #lfunc = lambda y: (M@y + q)
        lfunc = [M, q]
        subproblem = QNSTR_LCP1(lfunc, ldomain)
        y_ = subproblem.run(y_, lr = 1, epsilon = 1e-12, max_step = 1000, display = False)  
        #y_ = subproblem.run(y, kappa2, zeta1, zeta2, beta1, beta2, eta, nu, tau, 1e-12, epsilon_criteria, L, L1, lr, max_step, Gamma_ini, 
        #                        mode, normalized, gram_schmidt, epsilon_bar, warm_up, x)  
        return y_     
    
    
    def stage1_QNSTR(self, x, y, domaing, lr_list, num_cores, y_size, funcg):
        # Stage 1
        print('stage 2')
        ##################################################
        loop_num = range(0, len(funcg), 1)
        results = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel)(y[i*y_size:(i+1)*y_size], x, domaing[i], lr_list[i,0], i, funcg[i]) for i in loop_num)
    
        for i in loop_num:
            y[i*y_size:(i+1)*y_size] = results[i][0]    
            lr_list[i,0] = results[i][1]
        ############################################################

        return y, lr_list
    
    
    
    def stage1_QNSTR_LCP(self, x, y, domaing, index, lr_list, group_size, y_size, num_cores, funcg):
        # Stage 1
        print('stage 2')
        ##################################################
        #loop_num = range(0, len(y), group_size)
        #m = y.shape[0]
        loop_index = range(len(index)-1)
        results = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel_linear_block)(y[y_size*index[i]:y_size*index[i+1],:], x, domaing[index[i]:index[i+1]], lr_list[i,0], [index[i], index[i+1]], funcg[index[i]:index[i+1]]) for i in loop_index)

        for i in range(len(index)-1):
            y[y_size*index[i]:y_size*index[i+1]] = results[i]  

        ############################################################

        return y
    
    
    def run(self,
            x, 
            y,  
            max_step = 10000,
            epsilon = 1e-8,
            group_size=100, 
            num_cores=-1):
        
        if self.judge(self.funcg)==True:
            print("second stage is LCP") 
            m = y[0].shape[0]
            group_num = int(np.ceil(len(y)/group_size))
            loop_num = range(0, len(y), group_size)
            index = []
            for i in loop_num:
                index.append(i)
            index.append(len(y)) 
            lr = np.ones([group_num, 1])
            for i in range(len(index)-1):
                M_norm = []
                for j in range(index[i], index[i+1], 1):
                    M_norm_, _ = np.linalg.eigh(self.funcg[j][1].T@self.funcg[j][1])
                    M_norm.append(max(M_norm_))
                lr[i,0] = 0.5*(m/max(M_norm))**(1/2)
            lr_= 1. 
            #####################################
            for i in range(len(y)):
                if i == 0:
                    y_vector = y[i]
                else:
                    y_vector = np.concatenate([y_vector, y[i]],0)
            for i in range(max_step):
                y_vector = self.stage1_QNSTR_LCP(x, y_vector, self.domaing, index, lr, group_size, m, num_cores, self.funcg)
                #res = self.residual_LCP(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
                res = self.residual(x, y_vector, self.domainf, self.funcf)

                print('residual: ' +str(res))
                if res < epsilon:
                    print('converge!!!')
                    break

                print('stage 2')
                subproblem2 = QNSTR(self.funcf, self.domainf)
                x, lr_ = subproblem2.run(x, y_vector, lr = lr_, epsilon = 1e-12, max_step = 1000, display= False)         
            
        else:
            m = y[0].shape[0]
            for i in range(len(y)):
                if i == 0:
                    y_vector = y[i]
                else:
                    y_vector = np.concatenate([y_vector, y[i]],0)            
            lr = 1*np.ones([len(y),1])
            lr_= 1.
            #res = self.residual(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
            #print('residual: ' +str(res))
            for i in range(max_step):
                y_vector, lr = self.stage1_QNSTR(x, y_vector, self.domaing, lr, num_cores, m, self.funcg)
                #res = self.residual(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
                res = self.residual(x, y_vector, self.domainf, self.funcf)

                print('residual: ' +str(res))
                if res < epsilon:
                    print('converge!!!')
                    break

                print('stage 2')
                subproblem2 = QNSTR(self.funcf, self.domainf)
                x, lr_ = subproblem2.run(x, y_vector, lr = lr_, epsilon = 1e-12, max_step = 1000, display= False)         

                #t = 1e-3
                #x_ = proj(x - t*self.funcf(x,y), self.domainf)
                #x_bar = proj(x - t*self.funcf(x_,y), self.domainf)
                #x = x_bar

                #res = self.residual(x, y, self.domainf, self.domaing, self.funcf, self.funcg)
                #print('residual: ' +str(res))
                #if res<=epsilon:
                #    print('Converged !!!')
                #    break
    
        return x, y_vector

    

    

   


if __name__ == "__main__":
    n = 50
    m = 10
    N = 100

    A, B, T, M, q, q_, p = get_two_SLCP(N, n, m)
    #A, B, p, q, Q, T, a, a_0, b, b_0 = get_demo1(N, n, m)
    #A, B, p, q, Q, P, T, q_, a = get_demo2(N, n, m)

    for i in range(N):
        if i == 0:
            pB = p[i,0]*B[i,0]
        else:
            pB = np.concatenate([pB, p[i,0]*B[i,0]],1)
    
    
    domainf = [0.*np.ones([n,1]), 20*np.ones([n,1])]
    domaing = []
    for i in range(N):
        domaing.append([0.*np.ones([m,1]), np.inf*np.ones([m,1])])
        
    x_init = np.random.uniform(size=(n,1))
    x_init = np.clip(x_init, domainf[0],domainf[1])

    y_init = []
    for i in range(N):
        y_init.append(np.zeros([m,1]))

    def H(x,y):
        H1 = A@x + q + pB@y
        #for i in range(len(y)):
        #    H1 = H1 + p[i,0]*B[i,0]@y[i]
        return H1


    G = [lambda y,x, i=i: T[i,0]@x + M[i,0]@y + q_[i,0] for i in range(0, N)]
    #G = [lambda y,x, i=i: T[i,0]@x + 
    #     ((b[i,0].T@y + b_0[i,0]) * (2*Q[i,0]@y + a[i,0]) - b[i,0]*(y.T@Q[i,0]@y + a[i,0].T@y + a_0[i,0]))
    #     /((b[i,0].T@y + b_0[i,0])**2) for i in range(0, N)]
    #G = [lambda y,x, i=i: T[i,0]@x + (np.exp(-y.T@Q[i,0]@y)+a[i,0])*(P[i,0]@y+q_[i,0]) for i in range(0, N)]
    start_time = time.time()    
    demo = two_stage_LCP(
        funcf=H,
        funcg=G,
        domainf=domainf,
        domaing=domaing,
    )
    x_, y_ = demo.run(
        x=x_init,
        y=y_init,
        max_step = 10000,
        group_size = 2, 
        num_cores = -1
    )
    print('time cost: ' +str(time.time()-start_time))
    print(x_)