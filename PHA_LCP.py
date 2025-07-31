import pandas as pd
import numpy.matlib
import copy
import math
from scipy.linalg import block_diag
import time

from joblib import Parallel, delayed
import multiprocessing
import cProfile

from QNSTR_LCP1 import *
from generate_two_stage_LCP import *

class PHA_solver:


    def __init__(self, func, domain, prob):
        self.func = func
        self.domain = domain
        self.p = prob


    def residual(self, z):
        F = 0 
        for i in range(len(self.func)):
            F = F + self.p[i,0]*(self.func[i][:xdim,:]@z[i,0] + self.func[i][1][:xdim,:])
        q = z[0,0][:xdim,:] - F
        F1 = z[0,0][:xdim,:]- (np.maximum(self.domain[i][0][:xdim,:]-q,0) + q - np.maximum(q-self.domain[i][1][:xdim,:],0))
        res = np.linalg.norm(F1)**2
        for i in range(len(self.func)):
            q = z[i,0][xdim:,:] - self.func[i][0][xdim:,:]@z[i,0]-self.func[i][1][xdim:,:]
            F1 = z[i,0][xdim:,:]- (np.maximum(self.domain[i][0][xdim:,:]-q,0) + q - np.maximum(q-self.domain[i][1][xdim:,:],0))
            res = res + np.linalg.norm(F1)**2
        return res**(1/2)
    
    
    def residual1(self, z):
        F = 0 
        for i in range(len(self.func)):
            F = F + self.p[i,0]*(self.func[i][0][:xdim,:]@z[i,0] + self.func[i][1][:xdim,:])
        q = z[0,0][:xdim,:] - F
        F1 = z[0,0][:xdim,:]- (np.maximum(self.domain[i][0][:xdim,:]-q,0) + q - np.maximum(q-self.domain[i][1][:xdim,:],0))
        res = np.linalg.norm(F1)/(1+np.linalg.norm(z[0,0][:xdim,:]))
        for i in range(len(self.func)):
            q = z[i,0][xdim:,:] - self.func[i][0][xdim:,:]@z[i,0]-self.func[i][1][xdim:,:]
            F1 = z[i,0][xdim:,:]- (np.maximum(self.domain[i][0][xdim:,:]-q,0) + q - np.maximum(q-self.domain[i][1][xdim:,:],0))
            res = max(res, np.linalg.norm(F1)/(1+np.linalg.norm(z[i,0][xdim:,:])))
        return res    
    
    

    def update_parameter_parallel(self, z, W, hX, sigma, lr, domain, i, func):
        print('this is the ' +str(i+1) + 'th subproblem')
        size = func[0].shape[0]
        #lfunc = lambda x: (func(x) + sigma*x + W - sigma*hX)
        lfunc = [func[0]+sigma*np.eye(size), func[1] + W - sigma*hX]
        subproblem = QNSTR_LCP1(lfunc, domain)
        z_ = subproblem.run(z, epsilon = 1e-14, lr=lr, max_step = 10000, warm_up = True, display = False) 
        #del lfunc
        return z_     


    def stage1_QNSTR_LCP(self, z, W, hX, sigma, lr_list, num_cores):
        # Stage 1
        #print('stage 1')
        ##################################################
        loop_num = range(0, len(self.func), 1)
        results = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel)(z[i,0], W[i,0], hX[i,0], sigma, lr_list[i], self.domain[i], i, self.func[i]) for i in loop_num)

        step = 0
        for i in loop_num:
            z[i,0] = results[i]   
            #step = step + results[i][2]
        #print('average step cost in QNSTR: '+str(step/len(z)))
        return z
    
    def run(self,
            z,  
            xdim,
            ydim,
            max_step = 10000,
            epsilon = 1e-8,
            num_cores=-1):
        strart_time = time.time()
        sigma = 1.
        gamma =1.618
 
        lr = []
        for i in range(N):
            eigh,_ = np.linalg.eigh(self.func[i][0].T@self.func[i][0])
            lr.append(0.5/max(eigh)**(1/2))
        W = np.empty((N, 1), dtype = object)
        dim = xdim + ydim
        
        for i in range(N):
            W[i,0]=np.zeros([dim,1])
        
        X = copy.deepcopy(z)
        
        temp_x = 0
        for i in range(N): 
            temp_x = temp_x + X[i,0][:xdim,:]*self.p[i,0]
        
        hX = copy.deepcopy(X)
        for i in range(N):
            hX[i,0][:xdim,:] = temp_x
        
        
        for iters in range(max_step):
            X = self.stage1_QNSTR_LCP(X, W, hX, sigma, lr, num_cores)
            
    
            # Step 2: get implementable policy
            temp_x = 0
            for i in range(N):
                temp_x = temp_x + X[i,0][:xdim,:] * self.p[i,0]
            #hX = [np.copy(xi) for xi in X]
            hX = copy.deepcopy(X)
            for i in range(N):
                hX[i,0][:xdim,:] = temp_x
                W[i,0] = W[i,0] + gamma * sigma * (X[i,0] - hX[i,0])

            etaorg = self.residual1(hX)
            print('This is the '+str(iters) + ' th iteration---------time cost: ' +str(time.time()-start_time) + '--------------------------res: '+str(etaorg))

            if etaorg <= epsilon:
                break
                
        return hX




if __name__ == "__main__":
    
    n = 100
    m = 10
    N = 100
    
    A, B, T, M, q, q_, p = get_two_SLCP1(N, n, m)
    #A, B, p, q, Q, T, a, a_0, b, b_0 = get_demo1(N, n, m)
    #A, B, p, q, Q, P, T, q_, a = get_demo2(N, n, m)
    #M, q, p = get_two_SLCP_PHA(N, n, m)

    
    domainf = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]
    domaing = []
    for i in range(N):
        domaing.append([0.*np.ones([m,1]),np.inf*np.ones([m,1])])
    domain = []
    for i in range(N):
        domain.append([np.concatenate([domainf[0],domaing[i][0]],0), np.concatenate([domainf[1],domaing[i][1]],0)])
        
    z_init = np.empty((N, 1), dtype = object)
    for i in range(N):
        z_init[i,0] = np.random.uniform(size=(n+m,1))
        z_init[i,0] = np.clip(z_init[i,0], domain[i][0],domain[i][1])
    
    xdim = n
    ydim = m 
    
    AT = np.empty((N, 1), dtype = object)
    qT = np.empty((N, 1), dtype = object)
    for i in range(N):
        AT[i,0] = np.concatenate([np.concatenate([A, B[i,0]],1), np.concatenate([T[i,0],M[i,0]],1)],0)
        qT[i,0] = np.concatenate([q, q_[i,0]],0)
    #G = [lambda z, i=i: 
    #    AT[i,0]@z+ qT[i,0] for i in range(0, N)]
    #G = [lambda z, i=i: np.concatenate([
    #    A@z[0:xdim,:]+B[i,0]@z[xdim:,:]+q, T[i,0]@z[0:xdim,:] + M[i,0]@z[xdim:,:] + q_[i,0]],0) for i in range(0, N)]
    G = [[AT[i,0], qT[i,0]] for i in range(0,N)]
    
    start_time = time.time()    
    demo = PHA_solver(
        func=G,
        domain=domain,
        prob = p,
    )
    x_ = demo.run(
        z_init,
        xdim,
        ydim,
        max_step = 10000,
        num_cores = -1
    )
    print('time cost: ' +str(time.time()-start_time))
    print(x_)