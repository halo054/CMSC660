import pandas
import matplotlib.pyplot as plt
import numpy as np



Lsymm = np.genfromtxt('Lsymm.csv', delimiter=',')
bsymm = np.genfromtxt('bsymm.csv', delimiter=',')
M = np.genfromtxt('M.csv', delimiter=',')
#x_real = np.genfromtxt('x.csv', delimiter=',')
Lsymm_sliced = Lsymm[1:399,1:399]
bsymm_sliced = bsymm[1:399]



def CG_without_preconditioning(A,b):
    shape = np.shape(A)
    n = shape[0]
    
    x = np.random.rand(n,)
    
    r = np.matmul(A,x) 
    r = r.reshape((-1,))
    
    r = r - b
    
    p = -r
    k = 0
    
    residual_norm_list = []
    current_loss = np.linalg.norm(r)
    residual_norm_list.append(current_loss)
    
    while current_loss >= 1e-12:
#    while k <=1:
        alpha = np.matmul(np.transpose(r),r)
        denorm = np.matmul(np.transpose(p),A)
        
        denorm = np.matmul(denorm,p)
        alpha = alpha/denorm
        
        x = x + alpha * p
        
        new_r = r + alpha * np.matmul(A,p)
        
        
        beta = np.inner(new_r,new_r) / np.inner(r,r)
        p = -new_r + beta*p
        k+=1
        
        r = new_r
        current_loss = np.linalg.norm(r)
        residual_norm_list.append(current_loss)
    plt.plot(residual_norm_list)
    plt.title("CG_without_preconditioning")
    plt.yscale("log")
    
    return k,x

def CG_with_preconditioning(A,b,M):
    shape = np.shape(A)
    n = shape[0]
    
    x = np.random.rand(n,)
    r = np.matmul(A,x) 
    r = r.reshape((-1,))
    r = r - b
    
    y = np.linalg.solve(M,r)
    y = y.reshape((-1,))
    p = -y
    k = 0
    
    residual_norm_list = []
    current_loss = np.linalg.norm(r)
    residual_norm_list.append(current_loss)
    
    while current_loss >= 1e-12:
#    while k <=1:
        alpha = np.matmul(np.transpose(r),y)
        denorm = np.matmul(np.transpose(p),A)
        denorm = np.matmul(denorm,p)
        alpha = alpha/denorm
        
        x = x + alpha * p
        new_r = r + alpha * np.matmul(A,p)
        
        new_y = np.linalg.solve(M,new_r)
        
        beta = np.inner(new_r,new_y) / np.inner(r,y)
        p = -new_y + beta*p
        k+=1
        
        r = new_r
        current_loss = np.linalg.norm(r)
        residual_norm_list.append(current_loss)
        y = new_y
    plt.plot(residual_norm_list)
    plt.yscale("log")
    plt.title("CG_with_preconditioning")
    
    return k,x

k,x_sliced = CG_without_preconditioning(Lsymm_sliced,bsymm_sliced)
k_2,x_sliced_2 = CG_with_preconditioning(Lsymm_sliced,bsymm_sliced,M)

np.savetxt("x_sliced.csv", x_sliced, delimiter=",")
np.savetxt("x_sliced_2.csv", x_sliced_2, delimiter=",")