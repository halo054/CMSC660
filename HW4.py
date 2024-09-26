import numpy as np
import copy
import matplotlib.pyplot as plt

def RayleighQuotient(n):
    A = np.random.rand(n,n)
    AT = np.transpose(A)
    A = A + AT
    v = np.random.rand(n,1)
    vT = np.transpose(v)
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    k = 0
    mu = []
    #eigenvalues, eigenvectors = np.linalg.eig(A)
    #print("eigenvalue:",eigenvalues[:5])
    current_mu = np.matmul(vT, A)
    current_mu = np.matmul(current_mu, v)
    mu.append(current_mu)
    tol = 1e-12
    I = np.eye(n)
    res = abs(np.linalg.norm((np.matmul(A, v) - mu[k]* v)/mu[k]))
    print("k = ", k+1 ,": lam = ", mu[k][0][0], "    res = ", res)
    while res > tol:
        diag_mu = mu[k] * I
        #print("MU shape:", np.shape(mu))
        w = np.linalg.solve((A - diag_mu), v) 
        k += 1
        w_norm = np.linalg.norm(w)
        v = w / w_norm
        vT = np.transpose(v)
        current_mu = np.matmul(vT, A)
        current_mu = np.matmul(current_mu, v)
        mu.append(current_mu)
        
        res = abs(np.linalg.norm((np.matmul(A, v) - mu[k]* v)/mu[k]))
        print("k = ", k+1 ,": lam = ", mu[k][0][0], "    res = ", res)
    error = []
    #print(mu)
    for mus in mu:
        error.append(abs(mus[0][0] - mu[-1][0][0]))
    #print(error)
    decay_speed = []
    k = []
    for i in range(len(error)-2):
        #print("error[i]**3 = ",error[i]**3)
        #print()
        
        decay_speed.append( error[i+1] / (error[i]**3))
        k.append(i+1)
    #print(decay_speed)
    plt.scatter(k,decay_speed)
    plt.show()
    plt.close()

RayleighQuotient(50)    
RayleighQuotient(100)    
RayleighQuotient(200)    
RayleighQuotient(500) 
RayleighQuotient(1000)
RayleighQuotient(2000)  
RayleighQuotient(5000) 
        