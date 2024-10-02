import numpy as np
import copy 
import scipy

'''
A = np.random.rand(5,5)
AT = np.transpose(A)
A = A + AT
ori_A = copy.deepcopy(A)
print("Ori:")
print(A)
I = np.eye(5)

for i in range(5):
    value = A[i,i]
    for j in range(i,5):
        A[i,j] = A[i,j] / value
    for j in range(5):
        I[i,j] = I[i,j] / value
    for k in range(5):
        if k != i:
            value = A[k,i]
            for j in range(i,5):
                A[k,j] = A[k,j] - value*A[i,j]
            for j in range(0,5):
                I[k,j] = I[k,j] - value*I[i,j]
print("A:")
print(A)
print("I:")
print(I)
I_in_theory = np.matmul(ori_A,I)
print("A*A^-1:")
print(I_in_theory)
'''

'''
A = np.random.rand(5,5)
AT = np.transpose(A)
A = A + AT
ori_A = copy.deepcopy(A)
P, L, U = scipy.linalg.lu(A)
A = np.matmul(L,U)
ori_A = copy.deepcopy(A)
L_inverse = np.eye(5)
U_inverse = np.eye(5)


#print("Ori:")
#print(A)
print("L:\n",L)
#print("U:\n",U)



for i in range(5):
    for j in range(5-i-1):
        #print("index for value",j+i+1,i)
        value = L[j+i+1,i]
        #print(value)
        for k in range(i+1):
            #print("index for L_inverse",j+i+1,k)
            L_inverse[j+i+1,k] = L_inverse[j+i+1,k] - value * L_inverse[i,k]
            
            
            
            #print(L_inverse)
            #print()
            #print()
            #print()
        
#print("L_inverse:\n",L_inverse)
should_be_I = np.matmul(L,L_inverse)
#print("should_be_I:\n",should_be_I)


ori_U = copy.deepcopy(U)
for i in range(5):
    value = U[i,i]
    U_inverse[i,i] = U_inverse[i,i]/value
    for j in range(i,5):
        U[i,j] = U[i,j]/value

        
print("U:\n",U)
for i in range(5-1):
    for j in range(i+1):
        value = U[j,i+1]
        
        for k in range(5 - i -1):
            U[j,k+1+i] = U[j,k+1+i] - value* U[i+1,k+1+i]
            print("j,k+1+i",j,k+1+i)
            #print("U:\n",U)
            U_inverse[j,k+1+i] = U_inverse[j,k+1+i] - value* U_inverse[i+1,k+1+i]       
            #print("U_inverse:\n",U_inverse)
            #print()


#print()
#print()
#print("U:\n",U)
#print("U_inverse:\n",U_inverse)
#print("L_inverse:\n",L_inverse)

real_A_inverse = np.linalg.inv(ori_A)
print("real_A_inverse:\n",real_A_inverse)


A_inverse = np.zeros((5,5))

for i in range(5):
    for j in range(5):
        for k in range(5-i):
            A_inverse[i,j] += U_inverse[i,k+i]*L_inverse[k+i,j]




print("A_inverse:\n",A_inverse)


#should_be_I = np.matmul(ori_U,U_inverse)
#print("should_be_I:\n",should_be_I)
'''

def Cholesky(A):
    n = np.shape(A)
    n = n[0]
    L = np.zeros((n,n))
    for j in range(n):
        summation = 0
        for k in range(0,j):
            summation += L[j,k]**2
        flag = A[j,j] - summation
        if flag <= 0 :
            print("The matrix is not positive definite.")
            return False,-1
        #print("flag",j,j,"=",flag)
        L[j,j] = flag**(1/2)
        
        for i in range(j,n-1):
            summation = 0
            for k in range(0,j):
                summation += L[i+1,k]*L[j,k]
            L[i+1,j] = (A[i+1,j] - summation)/L[j,j]
    L2_norm = np.linalg.norm(L,2)
    return L,L2_norm


A = np.random.rand(100,100)
AT = np.transpose(A)
A1 = A + AT
A2 = np.matmul(AT,A)
ori_A1 = copy.deepcopy(A1)
ori_A2 = copy.deepcopy(A2)

#print("Ori:")
#print(A)
eig = np.linalg.eigvals(A1)
min_eig = min(eig)
print("Min EigenValue for A + AT: ",min_eig)
#print("All EigenValues:",eig)
#print("Min EigenValue:",min_eig)

L,L2_norm = Cholesky(A1)
if L2_norm != -1:
    real_L = np.linalg.cholesky(ori_A1)
    difference_L = real_L - L
        
        
    real_L_norm = np.linalg.norm(real_L,2)
    difference = real_L_norm - L2_norm
    print("Norm Difference:",difference)
print()

eig2 = np.linalg.eigvals(A2)
min_eig2 = min(eig2)
print("Min EigenValue for ATA: ",min_eig2)
L,L2_norm = Cholesky(A2)
if L2_norm != -1:
    real_L = np.linalg.cholesky(ori_A2)
    difference_L = real_L - L
        
        
    real_L_norm = np.linalg.norm(real_L,2)
    difference = real_L_norm - L2_norm
    print("Norm Difference:",difference)




























