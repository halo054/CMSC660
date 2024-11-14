
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol


def function(x):
    return 100 * (x[1]-x[0]**2)**2 + (1-x[0])**2

def gradient(x):
    g = np.zeros(2)
    g[0] = 400*(x[0]**3) - 400*x[0]*x[1] +2*x[0]-2
    g[1] = 200*x[1]-200*(x[0]**2)
    return g

def Hessian(x):
    hessian = np.zeros((2,2))
    hessian[0,0] = 1200*(x[0]**2) - 400*x[1] +2
    hessian[0,1] = -400*x[0]
    hessian[1,0] = -400*x[0]
    hessian[1,1] = 200
    return hessian

def cauchy_point(B,g,Delta):
    ng = np.linalg.norm(g)
    ps = -g*Delta/ng
    aux = np.dot(g,B @ g)
    if( aux <= 0 ):
        p = ps
    else:
        a = np.minimum(ng**3/(Delta*aux),1)
        p = ps*a
    return p

def New_H(Hk,xk_plus_1,xk,delta_fk_plus_1,delta_fk,size):#count and m are for reset
    I = np.eye(size)
    sk = xk_plus_1 - xk
    yk = delta_fk_plus_1 - delta_fk
    rho_k = 1/ np.inner(yk,sk)
    temp = I - rho_k * np.outer(sk,np.transpose(yk))
    temp2 = I - rho_k * np.outer(yk,np.transpose(sk))
    new_H = np.matmul(temp,Hk)
    new_H = np.matmul(new_H,temp2) + rho_k * np.outer(sk,np.transpose(sk))
    return new_H

tol = 1e-6 # stop iterations when ||grad f|| < tol
iter_max = 1000 # the maximal number of iterations
Delta_max = 5 # the max trust-region radius
Delta_min = 1e-12 # the minimal trust-region radius
Delta = 1 # the initial radius
eta = 0.1 # step rejection parameter
subproblem_iter_max = 5 # the max # of iteration for quadratic subproblems
tol_sub = 1e-1 # relative tolerance for the subproblem
rho_good = 0.75 # if rho > rho_good, increase the trust-region radius
rho_bad = 0.25 # if rho < rho_bad, decrease the trust-region radius


real_solution = np.zeros(2)
real_solution[0] = 1
real_solution[1] = 1
x1 = np.zeros(2)
x1[0] = 1.2
x1[1] = 1.2
x2 = np.zeros(2)
x2[0] = -1.2
x2[1] = 1
x = x2

f = function(x)
print(f)


g = gradient(x)
norm_g = np.linalg.norm(g)
print("Initially, f = ",f,", ||grad f|| = ",norm_g)
iter = 1

I = np.eye(np.size(x),dtype = float)
m = 20
count = m
size = len(x)
H = np.eye(size)
last_x = x
last_g = g
H_reset_flag = False


x_list = []
y_list = []
z_list = []
norm_list = []


direction = 0 # 0 for BFGS, 1 for newton
H_reset_flag = False
if direction == 0:
    dir = "BFGS Dogleg"
else:
    dir = "Newton"

while (norm_g > tol and iter < iter_max): 
    x_list.append(x[0])
    y_list.append(x[1])
    z_list.append(function(x))
    
    norm = np.linalg.norm(x-real_solution)
    norm_list.append(norm)
    
    
    #choose search direction
    g = gradient(x)
    if direction == 0:
        if count == m or H_reset_flag or np.linalg.norm(x - last_x) < 1e-12:
            H = np.eye(size)
            count = 0
            
        else:
            H = New_H(H,x,last_x,g,last_g,size)
            count += 1
        H_reset_flag = False
        B = np.linalg.inv(H)
        pb = -np.matmul(H,g)
        pb_norm = np.linalg.norm(pb)
        current_p = pb
        flag_boundary = 0
        if pb_norm > Delta:
            flag_boundary = 1
            B = np.linalg.inv(H)
            temp = np.dot(g,B @ g)
            pu = -np.inner(g,g)*g/temp
            pu_norm = np.linalg.norm(pu)
            if pu_norm >= Delta:
                pu = -Delta * g /np.linalg.norm(g)
                current_p = pu
                print("pu")
                
            else:
                pb_inner_pu = np.inner(pb,pu)
                a = pu_norm**2
                b = pb_inner_pu
                c = pb_norm**2
                alpha = Symbol('alpha')
                solution = solve(pu_norm**2 * (1-alpha)**2 + 2*alpha*(1-alpha) * pb_inner_pu + alpha**2*pb_norm**2 - Delta**2)
                #manual_solution = (2 * (a - b) + (4 * ( a - b )**2 - 4*(a- Delta**2)*(a-2*b+c))**(1/2)) / (2*a-4*b+2*c)
                alpha = (float)(max(solution))
                #print("manual_solution",manual_solution)
                print("dogleg")
                current_p = pu + (alpha)*(pb-pu)
                dogleg_norm=np.linalg.norm(current_p)
                #print("dogleg norm:", dogleg_norm, "Delta:",Delta)
        else: 
            print("pb")
            
        
        # assess the progress
        xnew = x + current_p
        fnew = function(xnew)
        gnew = gradient(xnew)
        B = np.linalg.inv(H)
        mnew = f + np.dot(g,current_p) + 0.5*np.dot(current_p,B @ current_p)
        #print("mnew = ", mnew)
        print("Estimate reduction = ",f - mnew+1e-14)
        rho = (f - fnew)/(f - mnew+1e-14)
        # adjust the trust region
        if( rho < rho_bad ):
            Delta = np.maximum(0.25*Delta,Delta_min)
        else:
            if(  rho > rho_good and flag_boundary == 1 ):
                Delta = np.minimum(Delta_max,2*Delta)
        # accept or reject step
        if( rho > eta ):  # accept step         
            last_x = x
            last_g = g
            x = xnew
            f = fnew
            g = gnew
            norm_g = np.linalg.norm(g)
            print(f'Accept: iter {iter}: f = {f:.10f}, |df| = {norm_g:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}')
        else:
            print(f'Reject: iter {iter}: f = {f:.10f}, |df| = {norm_g:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}')
            H_reset_flag = True

        iter = iter + 1
        print()
        continue

    B = Hessian(x)
        
    flag_boundary = 0 
    # check if B is SPD
    eval_min = np.amin(np.real(scipy.linalg.eig(B, b=None, left=False, right=False)))
    j_sub = 0
    if( eval_min > 0 ): # B is SPD: B = R'*R, R'*R*p = -g 
        p = scipy.linalg.solve(B,-g)
        p_norm = np.linalg.norm(p)
        if( p_norm > Delta ): # else: we are done with solbing the subproblem
            flag_boundary = 1        
    else:
        flag_boundary = 1
    if( flag_boundary == 1 ): # solution lies on the boundary
        lam_min = np.maximum(-eval_min,0.0)
        lam = lam_min + 1
        R = scipy.linalg.cholesky(B+lam*I,lower = False)
        flag_subproblem_success = 0;
        while( j_sub < subproblem_iter_max ):
            j_sub = j_sub + 1;
            p = scipy.linalg.solve_triangular(np.transpose(R),-g,lower = True)
            p = scipy.linalg.solve_triangular(R,p,lower = False)
            p_norm = np.linalg.norm(p)
            dd = np.absolute(p_norm - Delta)
            if( dd < tol_sub*Delta ):
                flag_subproblem_success = 1
                break
            q = scipy.linalg.solve_triangular(np.transpose(R),p,lower = True)
            q_norm = np.linalg.norm(q);
            dlam = ((p_norm/q_norm)**2)*(p_norm - Delta)/Delta
            lam_new = lam + dlam;
            if (lam_new > lam_min):
                lam = lam_new
            else:
                lam = 0.5*(lam + lam_min)
            R = scipy.linalg.cholesky(B+lam*I,lower = False)
        if( flag_subproblem_success == 0 ):
            p = cauchy_point(B,g,Delta)
    # assess the progress
    xnew = x + p
    fnew = function(xnew)
    gnew = gradient(xnew)
    mnew = f + np.dot(g,p) + 0.5*np.dot(p,B @ p)
    rho = (f - fnew)/(f - mnew+1e-14)
    # adjust the trust region
    if( rho < rho_bad ):
        Delta = np.maximum(0.25*Delta,Delta_min)
    else:
        if(  rho > rho_good and flag_boundary == 1 ):
            Delta = np.minimum(Delta_max,2*Delta)
    # accept or reject step
    if( rho > eta ):  # accept step          
        x = xnew
        f = fnew
        g = gnew
        norm_g = np.linalg.norm(g)
        print(f'Accept: iter {iter}: f = {f:.10f}, |df| = {norm_g:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}, j_sub = {j_sub}')
    else:
        print(f'Reject: iter {iter}: f = {f:.10f}, |df| = {norm_g:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}, j_sub = {j_sub}')
    iter = iter + 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
x_list.append(x[0])
y_list.append(x[1])
z_list.append(function(x))
print(f"Result: f = {f:.10f}, ||grad f|| = {norm_g:.6e}")


x_axis = np.linspace(-2, 2, 5000)
y_axis = np.linspace(-2, 2, 5000)
X, Y = np.meshgrid(x_axis, y_axis)
Z = 100 * (Y-X**2)**2 + (1-X)**2
plt.contour(X, Y, Z,levels=[0,1, 15, 30,50,100],colors = ['b','g','r','y','k','m'])
plt.colorbar() # Add a colorbar to show the values
plt.scatter(x_list,y_list)
plt.title('Contour Plot '+dir)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.rcParams.update({'font.size': 20})
plt.plot(norm_list,linewidth = 2)
plt.title('Norm '+dir)
plt.yscale("log")
plt.xlabel("Iteration #")
plt.ylabel("Norm")
plt.show()

