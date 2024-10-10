import numpy as np
import matplotlib.pyplot as plt
import copy

file = open("vectors.txt","r")
data = file.readlines()
Nlines = len(data)
print("Nlines = ",Nlines)

wfile = open("words_idx.txt","r")
words = wfile.readlines()
Nwords = len(words)
print("Nwords = ",Nwords)

Ndocs = int(Nlines/2)
A = np.zeros((Nwords,Ndocs))
Doc_idx = np.zeros((Ndocs,))
label = np.zeros((Ndocs,))
for j in range(Ndocs):
    Doc_idx = data[2*j]
    line = data[2*j+1]
    line = line.split()
    numbers = [eval(i) for i in line]
    label = numbers[0]
    w_idx = np.array(numbers[1::2])-1 # make indices start from 0
    w_count = numbers[2::2]
    A[w_idx,j] = 1       
# np.savetxt("Amatrix.csv", A, delimiter=",")






k = 10
W = np.random.rand(Nwords,k)
H = np.random.rand(k,Ndocs)
iter_max = 200
R_FroNorm = np.zeros(iter_max-1)

def NMF_PGD(k,W,H,iter_max,R_FroNorm,learning_rate= 7e-3):
    W_zero = np.zeros((Nwords,k))
    H_zero = np.zeros((k,Ndocs))
    
    
    R = A - np.matmul(W,H)    
    for iter in range(iter_max):
        Wnew = W + learning_rate*np.matmul(R,np.transpose(H))
        Wt = np.transpose(W)
        Hnew = H + learning_rate*np.matmul(Wt,R)
        
        H = np.maximum(Hnew,H_zero)
        W = np.maximum(Wnew,W_zero)
        R = A - np.matmul(W,H)
        if iter !=0:
            R_FroNorm[iter-1] = np.linalg.norm(R,'fro')
    print("NMF_PGD R_FroNorm",R_FroNorm[-1])


    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8,8))
    plt.plot(R_FroNorm)
    plt.title("NMF_PGD R_FroNorm")
    ind_list = []
       
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8,8))
    for j in range(k):
        ind = np.squeeze(np.argwhere(W[:,j] > 0.65))
        ind_copy = copy.deepcopy(ind)
        ind_list.append(ind_copy)
        print(W[ind,j])
        
        plt.plot(np.sort(W[:,j]))
        plt.title("Sorted Word Intensity of presence ")

        if len(ind.tolist()) > 0:
            for i in range(len(ind.tolist())):
                print(ind[i]+1,words[ind[i]])
    plt.show()
    


def NMF_HALS(k,W,H,iter_max,R_FroNorm):
    u = np.zeros((Nwords,1))
    v = np.zeros((1,Ndocs))
    R = A - np.matmul(W,H)
    
    
    index = 0
    epsilon =1e-12
    
    for iter in range(iter_max+4):
        if index == k:
            index = 0
        temp = (np.matmul(R,np.transpose(H))[:,index]) / (np.matmul(H,np.transpose(H))[index,index]+epsilon)
        u[:,0] = np.maximum(-W[:,index],temp)
        temp_u = temp
        for j in range(Nwords):
            W[j,index] = W[j,index] + u[j,0] 
        R = R - np.outer(u,H[index,:])
        
        temp = (np.matmul(np.transpose(W),R)[index,:]) / (np.matmul(np.transpose(W),W)[index,index]+epsilon)
        v[0,:] = np.maximum(-H[index,:],temp)
        temp_v = temp
        for j in range(Ndocs):
            H[index,j] = H[index,j] + v [0,j]
        R = R - np.outer(W[:,index] ,v )
        
        if iter >= 5:
            R_FroNorm[iter-5] = np.linalg.norm(R,'fro')    
        index +=1
    print("HALS R_FroNorm",R_FroNorm[-1])
    
    
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8,8))
    plt.plot(R_FroNorm)
    plt.title("NMF_HALS R_FroNorm")
    ind_list = []
       
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8,8))
    for j in range(k):
        ind = np.squeeze(np.argwhere(W[:,j] > 0.36))
        ind_copy = copy.deepcopy(ind)
        ind_list.append(ind_copy)
        print(W[ind,j])
        
        plt.plot(np.sort(W[:,j]))
        plt.title("Sorted Word Intensity of presence ")

        if len(ind.tolist()) > 0:
            for i in range(len(ind.tolist())):
                print(ind[i]+1,words[ind[i]])
    plt.show()
    


print("Question 1:")
NMF_PGD(k,W,H,iter_max,R_FroNorm)
print()
print()
print()


print("Question 2:")
NMF_HALS(k,W,H,iter_max,R_FroNorm)
print()
print()
print()


print("Question 3:")
u,s,vt = np.linalg.svd(A,full_matrices = False)
s10 = np.zeros((Ndocs,Ndocs))
for i in range(10):
    s10[i,i] = s[i]
A10 = np.matmul(u,s10)
A10 = np.matmul(A10,vt)
R_FroNorm_A10 = np.linalg.norm(A - A10,'fro') 
print("SVD A10 R_FroNorm",R_FroNorm_A10)












