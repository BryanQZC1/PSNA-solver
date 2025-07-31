import pandas as pd
import numpy.matlib
import copy
import math
from scipy.linalg import block_diag
import time

from joblib import Parallel, delayed
import multiprocessing
import cProfile

from QNSTR import *
from generate_two_stage_LCP import *

class PHA_solver:


    def __init__(self, func, domain, prob):
        self.func = func
        self.domain = domain
        self.p = prob


    def residual(self, z):
        F = 0 
        for i in range(len(self.func)):
            F = F + self.p[i,0]*self.func[i](z[i,0])[:xdim,:]
        q = z[0,0][:xdim,:] - F
        F1 = z[0,0][:xdim,:]- (np.maximum(self.domain[i][0][:xdim,:]-q,0) + q - np.maximum(q-self.domain[i][1][:xdim,:],0))
        res = np.linalg.norm(F1)**2
        for i in range(len(self.func)):
            q = z[i,0][xdim:,:] - self.func[i](z[i,0])[xdim:,:]
            F1 = z[i,0][xdim:,:]- (np.maximum(self.domain[i][0][xdim:,:]-q,0) + q - np.maximum(q-self.domain[i][1][xdim:,:],0))
            res = res + np.linalg.norm(F1)**2
        return res**(1/2)
    
    
    def residual1(self, z):
        F = 0 
        for i in range(len(self.func)):
            F = F + self.p[i,0]*self.func[i](z[i,0])[:xdim,:]
        q = z[0,0][:xdim,:] - F
        F1 = z[0,0][:xdim,:]- (np.maximum(self.domain[i][0][:xdim,:]-q,0) + q - np.maximum(q-self.domain[i][1][:xdim,:],0))
        res = np.linalg.norm(F1)/(1+np.linalg.norm(z[0,0][:xdim,:]))
        for i in range(len(self.func)):
            q = z[i,0][xdim:,:] - self.func[i](z[i,0])[xdim:,:]
            F1 = z[i,0][xdim:,:]- (np.maximum(self.domain[i][0][xdim:,:]-q,0) + q - np.maximum(q-self.domain[i][1][xdim:,:],0))
            res = max(res, np.linalg.norm(F1)/(1+np.linalg.norm(z[i,0][xdim:,:])))
        return res    
    
    

    def update_parameter_parallel(self, z, W, hX, sigma, domain, lr, i, func):
        print('this is the ' +str(i+1) + 'th subproblem')
 
        lfunc = lambda x: (func(x) + sigma*x + W - sigma*hX)
        subproblem = QNSTR(lfunc, domain)
        z_, lr = subproblem.run(z, epsilon = 1e-12, lr = lr, max_step = 5000, warm_up = True, display = False) 
        #del lfunc
        return z_, lr      


    def stage1_QNSTR(self, z, W, hX, sigma, lr_list, num_cores):
        # Stage 1
        #print('stage 1')
        ##################################################
        loop_num = range(0, len(z), 1)
        results = Parallel(n_jobs=num_cores)(delayed(self.update_parameter_parallel)(z[i,0], W[i,0], hX[i,0], sigma, self.domain[i], lr_list[i], i, self.func[i]) for i in loop_num)

        step = 0
        for i in loop_num:
            z[i,0] = results[i][0]    
            lr_list[i] = results[i][1]
            #step = step + results[i][2]
        #print('average step cost in QNSTR: '+str(step/len(z)))
        return z, lr_list
    
    def run(self,
            z,  
            xdim,
            ydim,
            max_step = 10000,
            epsilon = 1e-8,
            num_cores=-1):
        strart_time = time.time()
        sigma = 1.
        gamma = 1.618
 
        lr = []
        for i in range(N):
            lr.append(1.)

        W = np.empty((N, 1), dtype = object)
        dim = xdim + ydim
        
        for i in range(N):
            W[i,0]=np.zeros([dim,1])
        
        X = copy.deepcopy(z)
        X, lr = self.stage1_QNSTR(X, W, z, sigma, lr, num_cores)
        
        temp_x = 0
        for i in range(N): 
           temp_x = temp_x + X[i,0][:xdim,:]*self.p[i,0]
        
        hX = copy.deepcopy(X)
        for i in range(N):
            hX[i,0][:xdim,:] = temp_x
        
        
        for iters in range(max_step):
            X, lr = self.stage1_QNSTR(X, W, hX, sigma, lr, num_cores)
            
    
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
    N = 1
    
    #A, B, T, M, q, q_, p = get_two_SLCP1(N, n, m)
    A, B, p, q, Q, T, a, a_0, b, b_0 = get_demo1(N, n, m)
    A, B, p, q, Q, P, T, q_, a = get_demo2(N, n, m)
    #M, q, p = get_two_SLCP_PHA(N, n, m)

    
    domainf = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]
    domain = []
    for i in range(N):
        domain.append([np.concatenate([domainf[0],2.*np.ones([m,1])],0), np.concatenate([domainf[1],10*np.ones([m,1])],0)])
        
    z_init = np.empty((N, 1), dtype = object)
    for i in range(N):
        z_init[i,0] = np.random.uniform(size=(n+m,1))
        z_init[i,0] = np.clip(z_init[i,0], domain[i][0],domain[i][1])
    
    xdim = n
    ydim = m 
    '''
    AT = np.empty((N, 1), dtype = object)
    qT = np.empty((N, 1), dtype = object)
    for i in range(N):
        AT[i,0] = np.concatenate([np.concatenate([A, B[i,0]],1), np.concatenate([T[i,0],M[i,0]],1)],0)
        qT[i,0] = np.concatenate([q, q_[i,0]],0)
    G = [lambda z, i=i: 
        AT[i,0]@z+ qT[i,0] for i in range(0, N)]'''
    #G = [lambda z, i=i: np.concatenate([
    #    A@z[0:xdim,:]+B[i,0]@z[xdim:,:]+q, T[i,0]@z[0:xdim,:] + M[i,0]@z[xdim:,:] + q_[i,0]],0) for i in range(0, N)]

    
    #G = [lambda z, i=i: np.concatenate([
    #    A@z[0:xdim,:]+B[i,0]@z[xdim:,:]+q,
    #    T[i,0]@z[0:xdim,:] + ((b[i,0].T@z[xdim:,:] + b_0[i,0]) * (2*Q[i,0]@z[xdim:,:] + a[i,0]) - b[i,0]*(z[xdim:,:].T@Q[i,0]@z[xdim:,:] + a[i,0].T@z[xdim:,:] + a_0[i,0]))/((b[i,0].T@z[xdim:,:] + b_0[i,0])**2)],0) for i in range(0, N)]
    G = [lambda z, i=i: np.concatenate([
        A@z[0:xdim,:]+B[i,0]@z[xdim:,:]+q,
        T[i,0]@z[0:xdim,:] + (np.exp(-z[xdim:,:].T@Q[i,0]@z[xdim:,:])+a[i,0])*(P[i,0]@z[xdim:,:]+q_[i,0])],0) for i in range(0, N)]
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