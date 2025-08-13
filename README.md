# <font color=red>xxxxx</font> solver
<font color=red>xxxxx</font> Solver is a Python implementation of the algorithm for solving a class of two stage stochastic nonlinear variational inequalities.
This package provides a robust and efficient optimizer for solving two stage stochastic nonlinear variational inequality problems. It is implemented by the structure of PSNA with subrountine solving with a quasi-Newton subspace trust region (QNSTR) algorithm.

---

## Features
-  Projection semismooth Newton algorithm (PSNA): Efficient structure for solving two stage stochastic nonlinear variational inequality.
-  Quasi-Newton Subspace Trust Region (QNSTR) Algorithm: Efficiently solves nonmonotone VI problems with box constraints.
-  Flexible Loss Function: Supports user-defined loss functions for a wide range of applications.
-  Reproducible and Extensible: Designed for research and practical use in adversarial learning and related fields.

---

## Statement
This package provide an algorithm to solve:
\begin{equation}
\begin{split}
0&\in \mathbb{E}_{P}[F(x,y(\xi),\xi)]+\mathcal{N}_{X}(x),\\
0&\in G(x,y(\xi),\xi)\in\mathcal{N}_{Y(\xi)}(y(\xi)), \quad\quad\text{for almost every (a.e.) $\xi\in\Xi}\\
\end{\split}
\end{equation}
by Sample Average Approximation (SAA), the user need to provide the sample size $N$, form of $F:\mathbb{R}^{n}\times\underbrace{\mathbb{R}^{m}\times\cdots\mathbb{R}^{m}}_{N}$ and $G_{i}:\mathbb{R}^{n}\times\mathbb{R}^{m}$, the set of $X$ and $Y_{i}$ for $i=1,\cdots,N$.


## Installation
We recommend using uv for dependency management:
```bash
uv sync
```

---

## Usage Example
``` Python
from two_stage_NVI_optimizer.optimizer import two_stage_NVI

A, B, p, q, Q, P, T, q_, a = get_demo2(N, n, m)
domainf = [0.*np.ones([n,1]), np.inf*np.ones([n,1])]
domaing = []

for i in range(N):
    domaing.append([0.*np.ones([m,1]), np.inf*np.ones([m,1])])

x_init = np.random.uniform(size=(n,1))
x_init = np.clip(x_init, domainf[0],domainf[1])

y_init = []
for i in range(N):
    y_init.append(np.zeros([m,1]))

# stage 1 function
def H(x,y):
    H1 = A@x + q
    for i in range(len(y))
        H1 = H1 + p[i,0]*B[i,0]@y[i]
        return H1

# stage 2 functions
G = [lambda y,x, i=i: T[i,0]@x + (np.exp(-y.T@Q[i,0]@y)+a[i,0])*(P[i,0]@y+q_[i,0]) for i in range(0, N)]

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
    max_step = 10000,
    group_size = 1,
    num_cores = -1
    )
    
print('time cost: ' +str(time.time()-start_time))
print(x_)
```

---

## Citation
If you use this code or algorithm in your research, please cite the following paper:
```tex
    @article{Qiu_2024,
       title={A Quasi-Newton Subspace Trust Region Algorithm for Nonmonotone Variational Inequalities in Adversarial Learning over Box Constraints},
        volume={101},
        SSN={1573-7691},
        url={http://dx.doi.org/10.1007/s10915-024-02679-y},
        DOI={10.1007/s10915-024-02679-y},
        number={2},
        journal={Journal of Scientific Computing},
        publisher={Springer Science and Business Media LLC},
        author={Qiu, Zicheng and Jiang, Jie and Chen, Xiaojun},
        year={2024},
        month=oct }

   @article{wang2023solving,
      title={Solving two-stage stochastic variational inequalities by a hybrid projection semismooth Newton algorithm},
      author={Wang, Xiaozhou and Chen, Xiaojun},
      journal={SIAM Journal on Scientific Computing},
      volume={45},
      number={4},
      pages={A1741--A1765},
      year={2023},
      publisher={SIAM}
    }
   ```
