#import numpy as np
import autograd.numpy as np

def get_two_SLCP(N, n, m):
    A = np.random.normal(loc=0, scale=1, size=(n,n))
    A = A.T@A +np.eye(n)
    q = np.random.normal(loc=0, scale=1, size=(n,1))
    
    B = np.empty((N, 1), dtype = object)
    T = np.empty((N, 1), dtype = object)
    M = np.empty((N, 1), dtype = object)
    q_= np.empty((N, 1), dtype = object)
    p = np.empty((N, 1), dtype = object)
    
    for i in range(N):
        B[i,0] = np.random.normal(loc=0, scale=1, size=(n, m))
        T[i,0] = np.random.normal(loc=0, scale=1, size=(m, n))
        q_[i,0]= np.random.normal(loc=0, scale=1, size=(m, 1))
        p[i,0] = np.random.uniform(low=0, high=1, size=(1))
        M_ = np.random.normal(loc=0, scale=1, size=(m, m))
        M[i,0] = M_.T@M_# + np.eye(m)
    p = p/np.sum(p)
    #################
    return A, B, T, M, q, q_, p


def get_two_SLCP1(N, n, m):
    dim = n + m

    B = np.empty((N, 1), dtype = object)
    T = np.empty((N, 1), dtype = object)
    M = np.empty((N, 1), dtype = object)
    q_= np.empty((N, 1), dtype = object)
    p = np.empty((N, 1), dtype = object)

    # constant M11
    rank = int(np.ceil(3 * dim / 4))
    a = np.random.rand(rank, 1)
    v = [np.random.rand(dim, 1) for _ in range(rank)]
    M1 = np.zeros((dim, dim))
    for i in range(rank):
        M1 += a[i, 0] * v[i] @ v[i].T

    A = M1[:n,:n]

    for i in range(N):
        M2_i = np.random.randn(dim, dim)
        lM2 = np.tril(M2_i, -1)
        M2_i = np.diag(np.diag(M2_i)) + lM2 - lM2.T
        M2_i[:n, :n] = 0
        for j in range(n, dim):
            M2_i[j, j] = 0
        M2 = M1 + M2_i
        B[i,0] = M2[:n,n:]
        T[i,0] = M2[n:,:n]
        M[i,0] = M2[n:,n:]

    q = -np.random.rand(n, 1)
    for i in range(N):
        q_[i,0] = -np.random.rand(m, 1)

    p = np.random.uniform(low=0, high=1, size=(N,1))
    p = p/np.sum(p)

    return A, B, T, M, q, q_, p


def get_demo1(N, n, m):
    A = np.random.normal(loc=0, scale=1, size=(n,n))
    A = A.T@A +np.eye(n)
    q = np.random.normal(loc=0, scale=1, size=(n,1))

    B = np.empty((N, 1), dtype = object)
    p = np.empty((N, 1), dtype = object)
    
    a_0 = np.empty((N, 1), dtype = object)
    b_0 = np.empty((N, 1), dtype = object)
    Q = np.empty((N, 1), dtype = object)
    a = np.empty((N, 1), dtype = object)
    b = np.empty((N, 1), dtype = object)

    T = np.empty((N, 1), dtype = object)
    
    for i in range(N):
        c_0 = np.random.normal(1)# 生成 c_0
        d_0 = np.random.normal(1) # 生成 d_0
        
        Q_ = np.random.normal(size=(m,m)) 
        Q_ = Q_.T@Q_ + np.eye(m)
        c = np.random.uniform(size=(m,1))
        d = np.random.uniform(size=(m,1))
        e = np.ones([m,1])
        a_ = e + c
        b_ = e + d
        a_0_ = 1 + c_0
        b_0_ = 1 + d_0

        p[i,0] = np.random.uniform(low=0, high=1, size=(1))
        B[i,0] = np.random.normal(loc=0, scale=1, size=(n, m))

        Q[i,0] = Q_
        T[i,0] = np.random.normal(loc=0, scale=1, size=(m, n))
        a[i,0] = a_
        b[i,0] = b_
        a_0[i,0] = a_0_
        b_0[i,0] = b_0_
    p = p/np.sum(p)
    return A, B, p, q, Q, T, a, a_0, b, b_0


def get_demo2(N, n, m):
    A = np.random.normal(loc=0, scale=1, size=(n,n))
    A = A.T@A +np.eye(n)
    q = np.random.normal(loc=0, scale=1, size=(n,1))

    B = np.empty((N, 1), dtype = object)
    p = np.empty((N, 1), dtype = object)


    q_ = np.empty((N, 1), dtype = object)
    a = np.empty((N, 1), dtype = object)
    Q = np.empty((N, 1), dtype = object)
    P = np.empty((N, 1), dtype = object)


    T = np.empty((N, 1), dtype = object)
    

    for i in range(N):
        a[i,0] = np.random.uniform(1)
        #a[i,0] = np.random.uniform(1)
        x_sol = np.random.normal(size=(m,1))
        Q_ = np.random.normal(size=(m,m))
        x_sol = np.maximum(x_sol,0)
        Q[i,0] = Q_.T@Q_ + np.eye(m)
        P_ = np.random.normal(size=(m,m)) 
        P[i,0] = P_.T@P_
        q_[i,0] = (-P[i,0]@x_sol)*(x_sol>0)+(-P[i,0]@x_sol+np.random.uniform(size=(m,1)))*(x_sol==0)

        
        p[i,0] = np.random.uniform(low=0, high=1, size=(1))
        B[i,0] = np.random.normal(loc=0, scale=1, size=(n, m))

        T[i,0] = np.random.normal(loc=0, scale=1, size=(m, n))
    
    p = p/np.sum(p) 

    return A, B, p, q, Q, P, T, q_, a