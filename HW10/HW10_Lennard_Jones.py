# Steepest descent and Newton line-search methods applied to LJ7 cluster in 3D 
import numpy as np
import matplotlib.pyplot as plt
from LJhelpers import *

# Set up the initial configuration

# Four lical minima of LJ7:
# f1 = -16.50538417 Pentagonal bipyramid 
# f2 = -15.93504306 Capped octahedron 
# f3 = -15.59321094 Tricapped tetrahedron 
# f4 = -15.53306005 Bicapped trigonal bipyramid

# Options: model = 0,1,2,3, or 4.
# Model 0 corresponds to a random initialization.
# Models 1--4 set the system up close to the corresponding local minima
# listed above.




Na = 7 #the number of atoms
rstar = 2**(1/6) # argument of the minimum of the Lennard-Jones pair potential V(r) = r^(-12) - r^(-6)
tol = 1e-9 # stop iterations when ||grad f|| < tol
iter_max = 1000 # the maximal number of iterations
draw_flag = 1 # if draw_flag = 1, draw configuration at every iteration
# parameters for backtracking line search
c = 0.1;
rho = 0.9;


"""---------- Choose initialization ----------"""
model = 0
if( model > 0):
    Na = 7
xyz = initial_configuration(model,Na,rstar)
drawconf(xyz,0.5*rstar);

x = remove_rotations_translations(xyz)
drawconf(LJvector2array(x),0.5*rstar)
print("LJpot = ",LJpot(x))
# print(LJhess(x))


"""---------- Choose Algorithm ----------"""
# start minimization
# choose algorithm
# direction = 0: steepest descent
# direction = 1: Newton
# direction = 2: BFGS
direction = 2

f = LJpot(x)
g = LJgrad(x)
norm_g = np.linalg.norm(g)
print("Initially, f = ",f,", ||grad f|| = ",norm_g)

fvals = np.zeros(iter_max)
fvals[0] = f
ngvals = np.zeros(iter_max)
ngvals[0] = norm_g

iter = 1

m = 20
count = m
size = len(x)
H = np.eye(size)

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

while (norm_g > tol and iter < iter_max): 
    #choose search direction
    if( direction == 0): # steepest descent
        p = -g
        dir = "SD"
    elif( direction == 1): # Newton
        H = LJhess(x)
        p = np.linalg.solve(H,-g) 
        # if( np.dot(g,p) < 0 ): # descent direction
            # dir = "Newton"            
#         print(np.linalg.eigvals(H))
        spd = np.all(np.linalg.eigvals(H) > 0)
        if( spd ): # H is SPD, use Newton's direction
            p = np.linalg.solve(H,-g) 
            dir = "Newton"
        else: # use the steepest descent direction
            p = -g
            dir = "SD";
            
#TODO
    elif (direction == 2):
        delta_fk = LJgrad(x)
        if count == m:
            H = np.eye(size)
            count = 0
        elif np.linalg.norm(x - last_x) < 1e-12:
            count+=1
            p = -g
            dir = "SD";
            
        else:
            H = New_H(H,x,last_x,delta_fk,last_delta_fk,size)
            count += 1
        p = -np.matmul(H,delta_fk)
        last_x = x
        last_delta_fk = LJgrad(x)
        dir = "BFGS"
    
    else:
        print("direction is out of range")
        break
    # normalize the search direction if its length greater than 1
    norm_p = np.linalg.norm(p)
    if( norm_p > 1):
        p = p/norm_p
    # do backtracking line search along the direction p
    a = 1 # initial step length
    f_temp = LJpot(x + a*p)
    cpg = c*np.dot(p,g)
#     print("cpg = ",cpg,"f = ",f,"f_temp = ",f_temp)
    while( f_temp > f + a*cpg ): # check Wolfe's condition 1
        a = a*rho
        if( a < 1e-14 ):
            print("line search failed\n");
            iter = iter_max-1
            break
        f_temp = LJpot(x + a*p)        
#         print("f_temp = ",f_temp)
    x = x + a*p
    f = LJpot(x)
    g = LJgrad(x)
    norm_g = np.linalg.norm(g)
#     print("iter ",iter,": dir = ",dir,", f = ",f,", ||grad f|| = ",norm_g,", step length = ",a)
    print(f"iter {iter}: dir = {dir}, f = {f:.6f}, ||grad f|| = {norm_g:.6e}, step length = {a:.3e}")
    if( iter%100 == 0 ):
        # restore all coordinates
        xyz = LJvector2array(x)
        #drawconf(xyz,0.5*rstar)
    fvals[iter] = f
    ngvals[iter] = norm_g
    iter = iter + 1
print(f"Result: f = {f:.10f}, ||grad f|| = {norm_g:.6e}")
    
print("model:", model, "algorithm: " + dir)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(5,5))
plt.plot(np.arange(iter),fvals[0:iter],linewidth = 2)
plt.xlabel("Iteration #")
plt.ylabel("Function values")

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(5,5))
plt.plot(np.arange(iter),ngvals[0:iter],linewidth = 2)
plt.xlabel("Iteration #")
plt.ylabel("||grad f||")
plt.yscale("log")

plt.rcParams.update({'font.size': 20})
H = LJhess(x)
evals = np.sort(np.linalg.eigvals(H))
plt.figure(figsize=(5,5))
plt.scatter(np.arange(np.size(x)),evals,s = 20)
plt.xlabel("index")
plt.ylabel("Eigenvalues of the Hessian")
if( evals[0] > 0 ):
    plt.yscale("log")
















