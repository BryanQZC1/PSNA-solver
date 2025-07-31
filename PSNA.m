function [iter,iter_nt,rk,rk_two_stage,int_rk,time,data,flag,z0]=PSNA(DataS,H,q,l,u,z0,n,m,nu,epsilon,maxk,block_size)
%          
%This function solves two-stage SLCP by PSNA.  The second stage subproblem was solved by the "LCPsolver.p" solver written by Xudong Li
%
% Input: DataS,H :  the coefficient matrix for VI, where DataS with each submatrix stored separately
%                    in cell structure and H is a arrow-shaped big matrix of the problem     
%        q is the constant term;
%        l : the low bound for first-stage VI (LCP)
%        u : the upper bound for first-stage VI (LCP)
%        z0 : the initial point;
%        n : the dimension for the first stage 
%        m : the dimension for the second stage 
%        nu : sample size
%        epsilon : the stopping accuracy
%        maxk : the maximum iterations 
%        block_size : the number of block-implementation for solving the second stage problem
%        
%
% Output: iter : the number of iterations
%         iter_nt : the number of Newton iterations
%         rk : the residual for the one-stage formulation
%         rk_two_stage :  the residual for the two-stage VI
%         int_rk : the initial residual
%         flag : a indicator; flag=0 indicates solves the problem successfully
%         z0 : the approximate solution;


t0=clock;
iter=0;
eta=0.9;
alpha=0.015;
mak_proj=100; 
iter_nt=0;
x0=z0(1:n,1);
y0=z0(n+1:n+nu*m,1);


[V,y0]=second_stage_cell(DataS,x0,y0,n,m,nu,block_size,1);   % solve the second-stage problem
z0=[x0;y0];
residual_vector=H(1:n,:)*z0+q(1:n,1);                        
rk=norm(median([x0-l,x0-u,residual_vector],2));              % redudual for the one-stage formulation VI
int_rk=rk;


data=[0,rk];




for k=1:maxk

    if (rk<=epsilon)
         flag=0;
        break;
    end

    %eps=min(1,rk);
    eps = 0;                                                                                       % taking regularized parameter zero appears to be best for the problems tested here                                                                                        
    x_hat = pathlcp(H(1:n,1:n)+V+eps*eye(n,n),H(1:n,:)*[x0;y0]+q(1:n)-(H(1:n,1:n)+V)*x0,[],[],x0); % solve the one-stage VI via PATH solver   
    [V,y1]=second_stage_cell(DataS,x_hat,y0,n,m,nu,block_size,1);                                  % solve the second stage problem
    residual_vector=H(1:n,:)*[x_hat;y1]+q(1:n,1);                                                  
    rk1=norm(median([x_hat-l,x_hat-u,residual_vector],2));                                         % redudual for the one-stage formulation VI
    if rk1<=eta*rk                                                                                 % the Newton step is accepted
        x1=x_hat;
        z1=[x1;y1];
        rk=rk1;
        iter_nt=iter_nt+1;
        iter=iter+1;
        data=[data;iter,rk];
    else                                                                                            % perform the extragradient step
        count=0;
        for i=1:mak_proj 
            x_half=median([l,u,x0-alpha*(H(1:n,:)*z0+q(1:n,1))],2);
            [~,y_half]=second_stage_cell(DataS,x_half,y0,n,m,nu,block_size,0);
            x_hat=median([l,u,x0-alpha*(H(1:n,:)*[x_half;y_half]+q(1:n,1))],2);
            [~,y_hat]=second_stage_cell(DataS,x_hat,y_half,n,m,nu,block_size,0);         
            residual_vector=H(1:n,:)*[x_hat;y_hat]+q(1:n,1);
            rkj=norm(median([x_hat-l,x_hat-u, residual_vector],2));
            iter=iter+1;
            count=count+1;
            data=[data;iter,rkj];
            if rkj<=eta*rk
               [V,y_hat]=second_stage_cell(DataS,x_hat,y_hat,n,m,nu,block_size,1);
               x1=x_hat;
               y1=y_hat;
               z1=[x1;y1];
               rk=rkj;
               break;
            end
            
            if rkj<=epsilon
                x1=x_hat;
                y1=y_hat;
                z1=[x1;y1];
                rk=rkj;
                break;
            end
            
            if iter==maxk
               x1=x_hat;
               y1=y_hat;
               z1=[x1;y1];
               rk=rkj;
                break;
            end
            
 
            x0=x_hat;
            y0=y_hat;
            z0=[x0;y0];
        end
        if count==mak_proj&&rkj>eta*rk 
            flag=3;
            break;
        end
    end
    

   


    if rk>epsilon&& norm(z1-z0)<=1e-6
        flag=2;
        break;
    end



   
    z0=z1;
    x0=x1;
    y0=y1;
    

   

    if rk>epsilon&&iter==maxk
        flag=1;
        break;
    end
end

t1=clock;
time=etime(t1,t0);

residual_vector=H*z0+q;
L=zeros(n+nu*m,1);U=inf.*ones(n+nu*m,1);
rk_two_stage=norm(median([z0-L,z0-U,residual_vector],2));



end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [V,y,Time]=second_stage_cell(DataS,x0,y0,n,m,nu,block_size,inverse)
%
%this code solves the second stage problem of PSNA
%
% Input: DataS : the coefficient matrix for VI
%        x0 : current iteration point 
%        y0 : warm start for solving the second-stage subproblem
%        n : the dimension for the first stage 
%        m : the dimension for the second stage 
%        nu : sample size
%        block_size : the number of size for block-implementation in solving the second-stage problem
%        inverse : 1 or 0 with 1 computing the generalized Hessian while 0 is not.
%
% Output: V : the generalized Hessian of the second-stage solution function
%         y : the solution of the second-stage problem

    y=[];
    V=zeros(n,n);
    Time=0;
    epsilon=1e-12;
    block=ceil(nu/block_size);
    
    for i=1:block
        
        if i<block
            
            Mi=[];Bi=[];Ni=[];qi=[];
            for j=1:block_size
                index=(i-1)*block_size+j;
                Ti=DataS{index}.M;
                ppi=DataS{index}.p;
                Mi = blkdiag(Mi,Ti(n+1:n+m,n+1:n+m));
                Bi = [Bi,ppi.*Ti(1:n,n+1:n+m)];
                Ni=[Ni;Ti(n+1:n+m,1:n)];            
                qi=[qi;DataS{index}.q(n+1:n+m)+Ti(n+1:n+m,1:n)*x0];
            end
            

            x0i=y0(1+(i-1)*block_size*m:i*block_size*m,1);
            [yi,~] = LCPsolver(Mi, qi,x0i,[]);           %  LCP solved written by Xudong Li; this code is better than PATH solver for solving the problem generated here
            %yi = pathlcp(Mi,qi,[],[],x0i);              %  path solver
            y(1+(i-1)*block_size*m:i*block_size*m,1)=yi;

         
            t0=clock;
            if inverse
                ind=find(abs(yi)-epsilon>=0);
                I=sparse(1:block_size*m,1:block_size*m,ones(block_size*m,1), block_size*m, block_size*m );
                Di=sparse(ind,ind,ones(length(ind),1),block_size*m,block_size*m);
                Vi=Bi*(inv(I-Di+Di*Mi)*Di*Ni);
                V=V+Vi;
            end
            t1=clock;
            Time=etime(t1,t0)+Time;
            
        else
            ssn=nu-(i-1)*block_size;
             Mi=[];Bi=[];Ni=[];qi=[];
            for j=1:ssn
                index=(i-1)*block_size+j;
                Ti=DataS{index}.M;
                ppi=DataS{index}.p;
                Mi = blkdiag(Mi,Ti(n+1:n+m,n+1:n+m));
                Bi = [Bi,ppi.*Ti(1:n,n+1:n+m)];
                Ni=[Ni;Ti(n+1:n+m,1:n)];            
                qi=[qi;DataS{index}.q(n+1:n+m)+Ti(n+1:n+m,1:n)*x0];
            end
            

            x0i=y0(1+(i-1)*block_size*m:nu*m,1);
            [yi,~] = LCPsolver(Mi, qi,x0i,[]);
            %yi = pathlcp(Mi,qi,[],[],x0i);              %path solver
            y(1+(i-1)*block_size*m:nu*m,1)=yi;

         
            t0=clock;
            if inverse
                ind=find(abs(yi)-epsilon>=0);
                I=sparse(1:ssn*m,1:ssn*m,ones(ssn*m,1), ssn*m, ssn*m );
                Di=sparse(ind,ind,ones(length(ind),1),ssn*m,ssn*m);
                Vi=Bi*(inv(I-Di+Di*Mi)*Di*Ni);
                V=V+Vi;
            end
            t1=clock;
            Time=etime(t1,t0)+Time;
            
        end
            
    
    end
    
     V=-V;
     
end

