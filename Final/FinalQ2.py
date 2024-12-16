import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol

Rhoop = 3 # the radius of the hoop
r0 = 1 # the equilibrial length of the springs
kappa = 1 # the spring constant
Nnodes = 21
A = np.zeros((Nnodes,Nnodes),dtype = int) # spring adjacency matrix
# vertical springs
for k in range(3):
    A[k,k+4] = 1
for k in range(4,7):  
    A[k,k+5] = 1
for k in range(9,12):  
    A[k,k+5] = 1
for k in range(14,17):  
    A[k,k+4] = 1
# horizontal springs
for k in range(3,7):
    A[k,k+1] = 1
for k in range(8,12):  
    A[k,k+1] = 1
for k in range(13,17):  
    A[k,k+1] = 1
# symmetrize
Asymm = A + np.transpose(A)
# indices of nodes on the hoop
ind_hoop = [0,3,8,13,18,19,20,17,12,7,2,1]
Nhoop = np.size(ind_hoop)
# indices of free nodes (not attached to the hoop)
ind_free = [4,5,6,9,10,11,14,15,16]
Nfree = np.size(ind_free)
# list of springs
springs = np.array(np.nonzero(A))

Nsprings = np.size(springs,axis=1)
#print(springs)

# Initialization

# Initial angles for the nodes are uniformly distributed around the range of 2*pi
# startting from theta0 and going counterclockwise
theta0 = 2*np.pi/3
theta = theta0 + np.linspace(0,2*np.pi,Nhoop+1)
theta = np.delete(theta,-1)
# Initial positions
pos = np.zeros((Nnodes,2))
pos[ind_hoop,0] = Rhoop*np.cos(theta)
pos[ind_hoop,1] = Rhoop*np.sin(theta)
pos[ind_free,0] = np.array([-1.,0.,1.,-1.,0.,1.,-1.,0.,1.])
pos[ind_free,1] = np.array([1.,1.,1.,0.,0.,0.,-1.,-1.,-1.]) 

# Initiallize the vector of parameters to be optimized
vec = np.concatenate((theta,pos[ind_free,0],pos[ind_free,1]))

def draw_spring_system(pos,springs,R,ind_hoop,ind_free):
    # draw the hoop
    t = np.linspace(0,2*np.pi,200)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(8,8))
    plt.plot(R*np.cos(t),R*np.sin(t),linewidth = 5,color = 'red')
    # plot springs
    Nsprings = np.size(springs,axis=1)
    for k in range(Nsprings):
        j0 = springs[0,k]
        j1 = springs[1,k]
        plt.plot([pos[j0,0],pos[j1,0]],[pos[j0,1],pos[j1,1]],color = 'black',linewidth = 3)    
    # plot nodes
    plt.scatter(pos[ind_hoop,0],pos[ind_hoop,1],s = 300,color = 'crimson')
    plt.scatter(pos[ind_free,0],pos[ind_free,1],s = 300,color = 'black')
    plt.show()
# Draw the initial configuration of the spring system
draw_spring_system(pos,springs,Rhoop,ind_hoop,ind_free)

def compute_gradient(theta,pos,Asymm,r0,kappa,R,ind_hoop,ind_free):
    Nhoop = np.size(ind_hoop)
    g_hoop = np.zeros((Nhoop,)) # gradient with respect to the angles of the hoop nodes
    Nfree = np.size(ind_free)
    g_free = np.zeros((Nfree,2)) # gradient with respect to the x- and y-components of the free nodes
    for k in range(Nhoop):
        ind = np.squeeze(np.nonzero(Asymm[ind_hoop[k],:])) # index of the node adjacent to the kth node on the hoop
        rvec = pos[ind_hoop[k],:] - pos[ind,:] # the vector from that adjacent node to the kth node on the hoop
        rvec_length = np.linalg.norm(rvec) # the length of this vector
        # print(k,ind,ind_hoop[k],rvec)
        g_hoop[k] = (rvec_length - r0)*R*kappa*(rvec[0]*(-np.sin(theta[k])) + rvec[1]*np.cos(theta[k]))/rvec_length
    for k in range(Nfree):
        ind = np.squeeze(np.array(np.nonzero(Asymm[ind_free[k],:]))) # indices of the nodes adjacent to the kth free node
        Nneib = np.size(ind)
        for j in range(Nneib):
            rvec = pos[ind_free[k],:] - pos[ind[j],:] # the vector from the jth adjacent node to the kth free node 
            rvec_length = np.linalg.norm(rvec) # the length of this vector
            g_free[k,:] = g_free[k,:] + (rvec_length - r0)*R*kappa*rvec/rvec_length
    # return a single 1D vector
    return np.concatenate((g_hoop,g_free[:,0],g_free[:,1]))     

def Energy(theta,pos,springs,r0,kappa):
    Nsprings = np.size(springs,axis = 1)
    E = 0.
    for k in range(Nsprings):
        j0 = springs[0,k]
        j1 = springs[1,k]
        rvec = pos[j0,:] - pos[j1,:]
        rvec_length = np.linalg.norm(rvec)        
        E = E + kappa*(rvec_length - r0)**2
    E = E*0.5
    return E

def vec_to_pos(vec):
    theta = vec[:Nhoop]
    pos[ind_hoop,0] = Rhoop*np.cos(theta)
    pos[ind_hoop,1] = Rhoop*np.sin(theta)
    # positions of the free nodes
    pos[ind_free,0] = vec[Nhoop:Nnodes]
    pos[ind_free,1] = vec[Nnodes:] 
    return theta,pos

def gradient(vec):
    theta,pos = vec_to_pos(vec) 
    return compute_gradient(theta,pos,Asymm,r0,kappa,Rhoop,ind_hoop,ind_free)

def func(vec):
    theta,pos = vec_to_pos(vec) 
    return Energy(theta,pos,springs,r0,kappa)



def Gradient_decent(vec):
    step_size = 0.001
    iter_max = 15000
    energy = func(vec)
    energy_list = [energy]
    #print(energy)
    grad_norm_list = []
    iter = 1
    while iter <= iter_max:
        grad = gradient(vec)
        grad_norm = np.linalg.norm(grad)
        grad_norm_list.append(grad_norm)
        if grad_norm > 1:
            grad = grad/grad_norm
        vec = vec - step_size * grad
        energy = func(vec)
        energy_list.append(energy)
        #print("iter = ",iter,"energy=",energy)
        iter += 1
        
        
    print("Gradient_decent")
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(iter_max),energy_list[0:iter_max],linewidth = 2)
    plt.xlabel("Iteration #")
    plt.ylabel("Energy values")
    

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(iter_max),grad_norm_list[0:iter_max],linewidth = 2)
    plt.xlabel("Iteration #")
    plt.ylabel("||grad f||")
    plt.yscale("log")    
    
    plt.show()
    
    return vec




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

def Trusted_region(vec):
    iter = 0
    iter_max = 10000
    tol = 1e-6
    Delta_max = 5 # the max trust-region radius
    Delta_min = 1e-12 # the minimal trust-region radius
    Delta = 1 # the initial radius
    eta = 0.1 # step rejection parameter
    rho_good = 0.75 # if rho > rho_good, increase the trust-region radius
    rho_bad = 0.25 # if rho < rho_bad, decrease the trust-region radius
    
    
    
    m = 20
    count = m
    grad = gradient(vec)
    grad_norm = np.linalg.norm(grad)
    f = func(vec)
    
    last_vec = vec
    last_grad = grad
    size = np.size(vec)
    I = np.eye(size,dtype = float)
    
    H_reset_flag = False
    dir = "BFGS Dogleg"
    energy_list = []
    energy_list.append(f)
    grad_norm_list = [grad_norm]
    grad_norm_list.append(f)
    while (grad_norm > tol and iter < iter_max): 
        #choose search direction
        grad = gradient(vec)
        if True:
        
            if count == m or H_reset_flag or np.linalg.norm(vec - last_vec) < 1e-12:
                H = np.eye(size)
                count = 0
                
            else:
                H = New_H(H,vec,last_vec,grad,last_grad,size)
                count += 1
            H_reset_flag = False
            B = np.linalg.inv(H)
            pb = -np.matmul(H,grad)
            pb_norm = np.linalg.norm(pb)
            current_p = pb
            flag_boundary = 0
            if pb_norm > Delta:
                flag_boundary = 1
                B = np.linalg.inv(H)
                temp = np.dot(grad,B @ grad)
                pu = -np.inner(grad,grad)*grad/temp
                pu_norm = np.linalg.norm(pu)
                if pu_norm >= Delta:
                    pu = -Delta * grad /np.linalg.norm(grad)
                    current_p = pu
                    print("pu")
                    
                else:
                    pb_inner_pu = np.inner(pb,pu)
                    #a = pu_norm**2
                    #b = pb_inner_pu
                    #c = pb_norm**2
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
            vecnew = vec + current_p
            fnew = func(vecnew)
            gradnew = gradient(vecnew)
            B = np.linalg.inv(H)
            mnew = f + np.dot(grad,current_p) + 0.5*np.dot(current_p,B @ current_p)
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
                
                last_vec = vec
                last_grad = grad
                vec = vecnew
                f = fnew
                grad = gradnew
                grad_norm = np.linalg.norm(grad)
                energy_list.append(f)
                grad_norm_list.append(grad_norm)
                #print(f'Accept: iter {iter}: f = {f:.10f}, |df| = {grad_norm:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}')
            else:
                #print(f'Reject: iter {iter}: f = {f:.10f}, |df| = {grad_norm:.4e}, rho = {rho:.4e}, Delta = {Delta:.4e}')
                H_reset_flag = True
            
            iter = iter + 1
            print()
    print("BFGS Dogleg")
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(len(energy_list)),energy_list,linewidth = 2)
    plt.xlabel("Iteration #")
    plt.ylabel("Energy values")
    

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(len(grad_norm_list)),grad_norm_list,linewidth = 2)
    plt.xlabel("Iteration #")
    plt.ylabel("||grad f||")
    plt.yscale("log")    
    
    plt.show()
    return vec
        
        


vec = Gradient_decent(vec)
theta,pos = vec_to_pos(vec)
print("Positions of the nodes:")
print(pos)
print()
print("Resulting energy:")
print(func(vec))
print()
print("Norm of the gradient:")
grad = gradient(vec)
grad_norm = np.linalg.norm(grad)
print(grad_norm)
print()
draw_spring_system(pos,springs,Rhoop,ind_hoop,ind_free)


vec = Trusted_region(vec)
theta,pos = vec_to_pos(vec)
draw_spring_system(pos,springs,Rhoop,ind_hoop,ind_free)
print("Positions of the nodes:")
print(pos)
print()
print("Resulting energy:")
print(func(vec))
print()
print("Norm of the gradient:")
grad = gradient(vec)
grad_norm = np.linalg.norm(grad)
print(grad_norm)
print()




















    