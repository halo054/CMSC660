import numpy as np
import matplotlib.pyplot as plt


tol = 1e-9 # stop iterations when ||grad f|| < tol
iter_max = 1000 # the maximal number of iterations
draw_flag = 1 # if draw_flag = 1, draw configuration at every iteration
# parameters for backtracking line search
c = 0.1;
rho = 0.9;



"""---------- Choose Algorithm ----------"""
# start minimization
# choose algorithm
# direction = 0: steepest descent
# direction = 1: Newton
# direction = 2: BFGS
direction = 2




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


solution = np.zeros(2)
solution[0] = 1
solution[1] = 1
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


fvals = np.zeros(iter_max)
fvals[0] = f
ngvals = np.zeros(iter_max)
ngvals[0] = norm_g

iter = 1

m = 5
count = m
size = len(x)
H = np.eye(size)
last_x = x
last_delta_fk = gradient(x)

step_length = []
x_list = []
y_list = []
z_list = []

norm_list = []

while (norm_g > tol and iter < iter_max): 
    x_list.append(x[0])
    y_list.append(x[1])
    z_list.append(function(x))
    
    norm = np.linalg.norm(x-solution)
    norm_list.append(norm)
    
    
    #choose search direction
    if( direction == 0): # steepest descent
        p = -g
        dir = "SD"
    elif( direction == 1): # Newton
        hessian = Hessian(x)
        p = np.linalg.solve(hessian,-g) 
        # if( np.dot(g,p) < 0 ): # descent direction
            # dir = "Newton"            
#         print(np.linalg.eigvals(H))
        spd = np.all(np.linalg.eigvals(hessian) > 0)
        if( spd ): # H is SPD, use Newton's direction
            p = np.linalg.solve(hessian,-g) 
            dir = "Newton"
        else: # use the steepest descent direction
            p = -g
            dir = "SD";
            

    elif (direction == 2):
        delta_fk = gradient(x)
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
        last_delta_fk = gradient(x)
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
    f_temp = function(x + a*p)
    cpg = c*np.dot(p,g)
#     print("cpg = ",cpg,"f = ",f,"f_temp = ",f_temp)
    while( f_temp > f + a*cpg ): # check Wolfe's condition 1
        a = a*rho
        if( a < 1e-14 ):
            print("line search failed\n");
            iter = iter_max-1
            break
        f_temp = function(x + a*p)        
#         print("f_temp = ",f_temp)
    x = x + a*p
    f = function(x)
    g = gradient(x)
    norm_g = np.linalg.norm(g)
#     print("iter ",iter,": dir = ",dir,", f = ",f,", ||grad f|| = ",norm_g,", step length = ",a)
    print(f"iter {iter}: dir = {dir}, f = {f:.6f}, ||grad f|| = {norm_g:.6e}, step length = {a:.3e}")

    fvals[iter] = f
    ngvals[iter] = norm_g
    iter = iter + 1
    step_length.append(a)
x_list.append(x[0])
y_list.append(x[1])
z_list.append(function(x))
print(f"Result: f = {f:.10f}, ||grad f|| = {norm_g:.6e}")


plt.rcParams.update({'font.size': 20})
plt.plot(step_length,linewidth = 2)
plt.title('Step length '+dir)
plt.xlabel("Iteration #")
plt.ylabel("Step length")
plt.show()

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
plt.xlabel("Iteration #")
plt.ylabel("Norm")
plt.show()







