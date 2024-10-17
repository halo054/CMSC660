import pandas
import matplotlib.pyplot as plt
import numpy as np
NumpyCSV = np.genfromtxt('MovieRankingData2024.csv', delimiter=',')
NumpyCSV[103,18] = 4.5
NumpyCSV = NumpyCSV[1:,2:]
shape = np.shape(NumpyCSV)
for i in range(shape[0]):
    for j in range(shape[1]):
        if not np.isnan(NumpyCSV[i,j]) :
            if NumpyCSV[i,j] > 5:
                NumpyCSV[i,j] = NumpyCSV[i,j]/2
    
shape = np.shape(NumpyCSV)

omega = []
#print(NumpyCSV[2,6])


for i in range(shape[0]):
    for j in range(shape[1]):
        if not np.isnan(NumpyCSV[i,j]) :
            omega.append([i,j])





def F_norm_completion(rank,penalty_lambda,iterations,shape = shape, data = NumpyCSV,omega = omega):
    n = shape[0]
    d = shape[1]
    X = np.random.rand(n,rank)
    YT = np.random.rand(rank,d)
    Y = np.transpose(YT)
    M = np.matmul(X,YT)
    R_Frob_norm = []
    
    i = 0
    j = 0
    
    for iter in range(iterations):
        if i == n:
            i = 0
        if j == d:
            j = 0
     
        columns = []
        for pairs in omega:
            if pairs[0] == i:
                columns.append(pairs[1])
         
        YT_omega_i = np.random.rand(rank,len(columns))
        a_omega_i = np.zeros((len(columns),1))
        I = np.eye(rank)
        lambda_I = penalty_lambda * I
        
        for index in range(len(columns)):
            YT_omega_i[:,index] = YT[:,columns[index]]
            a_omega_i[index,0] = data[i,columns[index]]
            
        Y_omega_i = np.transpose(YT_omega_i)    
        current_column = np.linalg.inv( np.matmul(YT_omega_i,Y_omega_i) + lambda_I )
        current_column = np.matmul(current_column,YT_omega_i)
        current_column = np.matmul(current_column,a_omega_i)
        for index in range(rank):
            X[i,index] = current_column[index,0]
        i+=1
        
        
        rows = []
        for pairs in omega:
            if pairs[1] == j:
                rows.append(pairs[0])
        X_omega_j = np.random.rand(len(rows),rank)
        a_omega_j = np.zeros((len(rows),1))
        I = np.eye(rank)
        lambda_I = penalty_lambda * I
        
        for index in range(len(rows)):
            X_omega_j[index,:] = X[rows[index],:]
            a_omega_j[index,0] = data[rows[index],j]
            
        XT_omega_j = np.transpose(X_omega_j)    
        current_column = np.linalg.inv( np.matmul(XT_omega_j,X_omega_j) + lambda_I )
        current_column = np.matmul(current_column,XT_omega_j)
        current_column = np.matmul(current_column,a_omega_j)
        
        for index in range(rank):
            YT[index,j] = current_column[index,0]
        j+=1
        
        
        loss = 0
        M = np.matmul(X,YT)
        for pairs in omega:
            loss += (data[pairs[0],pairs[1]] - M[pairs[0],pairs[1]]) ** 2 
        
        loss += (np.linalg.norm(X,'fro') ** 2 + np.linalg.norm(YT,'fro') ** 2) * penalty_lambda
        
        loss /=2
        #print("loss:",loss)
        R_Frob_norm.append(loss)
        
    plt.plot(R_Frob_norm)
    string = "rank = "+str(rank)+"    lambda = "+str(penalty_lambda) + " final loss = "+ str(R_Frob_norm[-1])
    plt.title(string)
    plt.yscale("log")
    plt.show()
    return X,YT




'''
X,YT = F_norm_completion(1,0.1,2000)
M_1 = np.matmul(X,YT)

X,YT = F_norm_completion(2,0.1,2000)
M_2 = np.matmul(X,YT)

X,YT = F_norm_completion(3,0.1,2000)
M_3 = np.matmul(X,YT)

X,YT = F_norm_completion(4,0.1,2000)
M_4 = np.matmul(X,YT)

X,YT = F_norm_completion(5,0.1,2000)
M_5 = np.matmul(X,YT)

X,YT = F_norm_completion(6,0.1,2000)
M_6 = np.matmul(X,YT)

X,YT = F_norm_completion(7,0.1,2000)
M_7 = np.matmul(X,YT)


X,YT = F_norm_completion(1,1,2000)
M_8 = np.matmul(X,YT)

X,YT = F_norm_completion(2,1,2000)
M_9 = np.matmul(X,YT)

X,YT = F_norm_completion(3,1,2000)
M_10 = np.matmul(X,YT)

X,YT = F_norm_completion(4,1,2000)
M_11 = np.matmul(X,YT)

X,YT = F_norm_completion(5,1,2000)
M_12 = np.matmul(X,YT)

X,YT = F_norm_completion(6,1,2000)
M_13 = np.matmul(X,YT)

X,YT = F_norm_completion(7,1,2000)
M_14 = np.matmul(X,YT)


X,YT = F_norm_completion(1,10,2000)
M_15 = np.matmul(X,YT)

X,YT = F_norm_completion(2,10,2000)
M_16 = np.matmul(X,YT)

X,YT = F_norm_completion(3,10,2000)
M_17 = np.matmul(X,YT)

X,YT = F_norm_completion(4,10,2000)
M_18 = np.matmul(X,YT)

X,YT = F_norm_completion(5,10,2000)
M_19 = np.matmul(X,YT)

X,YT = F_norm_completion(6,10,2000)
M_20 = np.matmul(X,YT)

X,YT = F_norm_completion(7,10,2000)
M_21 = np.matmul(X,YT)



'''




def s_lambda(singular_value, penalty_lambda):
    return max(singular_value - penalty_lambda,0)

def S_lambda(matrix,penalty_lambda):
    u,s,vt = np.linalg.svd(matrix,full_matrices = False)
    for i in range(len(s)):
        s[i] = s_lambda( s[i], penalty_lambda)
    I = np.eye(len(s))
    s = s * I
    A = np.matmul(u,s)
    A = np.matmul(A,vt)
    return A


def N_norm_completion(penalty_lambda, iterations, shape = shape, data = NumpyCSV,omega = omega):
    n = shape[0]
    d = shape[1]
    M = np.random.rand(n,d)
    N_Frob_norm = []
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if  np.isnan(data[i,j]) :
                data[i,j] = 0
    
    for iter in range(iterations):
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if  data[i,j] != 0 :
                    M[i,j] += data[i,j] - M[i,j]
        M = S_lambda(M,penalty_lambda)

        loss = 0
        for pairs in omega:
            loss += (data[pairs[0],pairs[1]] - M[pairs[0],pairs[1]]) ** 2 
        loss += np.linalg.norm(M,'nuc')*penalty_lambda
        N_Frob_norm.append(loss)
    plt.plot(N_Frob_norm)
    string = "lambda = "+str(penalty_lambda) + "    final loss = "+ str(N_Frob_norm[-1])
    plt.title(string)
    plt.yscale("log")
    plt.show()
    return M




M_22 = N_norm_completion(0.1, 2000)
M_23 = N_norm_completion(0.5, 500)
M_24 = N_norm_completion(1, 200)
M_25 = N_norm_completion(10, 200)






A = np.zeros((2,2))
A[0,0] = 0.5

z_list = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,]



epsilon = 3e-3
for index in range(len(z_list)):


    z = z_list[index]
    A[1,1] = z
    x_list = []
    y_list = []
    x_det_list = []
    y_det_list = []

    x = -2
    while x < 2:
        A[0,1] = x
        
        y = -2
        while y < 2:
            A [1,0] = y
            
            nuc_norm = np.linalg.norm(A,'nuc')
            det = np.linalg.det(A)
            if abs(nuc_norm-1) <= epsilon:
                x_list.append(x)
                y_list.append(y)
            if abs(det) <= epsilon:
                x_det_list.append(x)
                y_det_list.append(y)
                
            y += 0.005
        x+=0.005
    
        
    plt.scatter(x_list,y_list)
    plt.scatter(x_det_list,y_det_list)
    string = "Z = " + str(z)
    
    plt.title(string)
    plt.show()
































