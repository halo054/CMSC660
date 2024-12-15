import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#from Levenberg_Marquardt import LevenbergMarquardt
import scipy
import copy

        
def process_train_dateset(NPCA,d1,d2,temp_Ntrain_1 , temp_Ntrain_7 ,temp_imgs_train_1 ,temp_imgs_train_7 ):

    dd = d1*d2
    X1 = np.zeros((temp_Ntrain_1,dd))
    X7 = np.zeros((temp_Ntrain_7,dd))
    for j in range(temp_Ntrain_1):
        img = np.squeeze(temp_imgs_train_1[:,:,j])
        X1[j,:] = np.reshape(img,(dd,))
    for j in range(temp_Ntrain_7):
        img = np.squeeze(temp_imgs_train_7[:,:,j])
        X7[j,:] = np.reshape(img,(dd,))
    X = np.concatenate((X1,X7),axis = 0)
    print(np.shape(X))
    U,S,Vtrans = np.linalg.svd(X,full_matrices = False)
    print(f"U: {np.shape(U)}; S: {np.shape(S)}; Vtrans: {np.shape(Vtrans)}")
    V = np.transpose(Vtrans)
    Xtrain = np.matmul(X,V[:,:NPCA])
    print(f"Xtrain: {np.shape(Xtrain)}")
    return Xtrain,X1,X7,U,S,V

def draw_data_projection(X1,X7,V,NPCA):
    
    # Plot train data projected onto the first three PCAs
    X1_3pca = np.matmul(X1,V[:,:NPCA])
    X7_3pca = np.matmul(X7,V[:,:NPCA])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.rcParams.update({'font.size': 16})
    plt.scatter(X1_3pca[:,0],X1_3pca[:,1],X1_3pca[:,2],color = 'red',label = "1")
    plt.scatter(X7_3pca[:,0],X7_3pca[:,1],X7_3pca[:,2],color = 'blue',label = "7")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

def process_test_dateset(NPCA,d1,d2,temp_Ntest_1, temp_Ntest_7,temp_imgs_test_1,temp_imgs_test_7,V):
# Prepare the test set
    dd = d1*d2
    X1test = np.zeros((temp_Ntest_1,dd))
    X7test = np.zeros((temp_Ntest_7,dd))
    for j in range(temp_Ntest_1):
        img = np.squeeze(temp_imgs_test_1[:,:,j])
        X1test[j,:] = np.reshape(img,(dd,))
    for j in range(temp_Ntest_7):
        img = np.squeeze(temp_imgs_test_7[:,:,j])
        X7test[j,:] = np.reshape(img,(dd,))
    Xtest = np.concatenate((X1test,X7test),axis = 0)
    Xtest = np.matmul(Xtest,V[:,:NPCA])
    print(f"Xtest: {np.shape(Xtest)}")
    return Xtest,X1test,X7test

def change_label(temp_Ntrain_1 ,temp_Ntrain_7 ,temp_Ntest_1 ,temp_Ntest_7,Ntrain_1,Ntest_1):
    # prepare labels for the train and test sets: ones have label 1, sevens have label -1
    Ntrain = temp_Ntrain_1 + temp_Ntrain_7
    Ntest = temp_Ntest_1 + temp_Ntest_7
    lbl_train = np.ones((Ntrain,))
    lbl_train[Ntrain_1:] = -1
    lbl_test = np.ones((Ntest,))
    lbl_test[Ntest_1:] = -1
    return lbl_train,lbl_test





# Read mnist data from the mat file
mnist_data = scipy.io.loadmat("mnist2.mat")
imgs_train = mnist_data['imgs_train']
imgs_test = mnist_data['imgs_test']
labels_train = np.squeeze(mnist_data['labels_train'])
labels_test = np.squeeze(mnist_data['labels_test'])
print(np.shape(imgs_train))
print(np.shape(imgs_test))
print(np.shape(labels_train))
print(np.shape(labels_test))
print(labels_train[:100])
d1,d2,N = np.shape(imgs_train)


# Select images of 1 and 7
Itrain_3 = np.where(labels_train == 3)
Itrain_8 = np.where(labels_train == 8)
Itrain_9 = np.where(labels_train == 9)

imgs_train_3 = np.squeeze(imgs_train[:,:,Itrain_3])
imgs_train_8 = np.squeeze(imgs_train[:,:,Itrain_8])
imgs_train_9 = np.squeeze(imgs_train[:,:,Itrain_9])

Ntrain_3 = np.size(Itrain_3)
Ntrain_8 = np.size(Itrain_8)
Ntrain_9 = np.size(Itrain_9)

print(f"Ntrain_3 = {Ntrain_3}, Ntrain_8 = {Ntrain_8}, Ntrain_9 = {Ntrain_9}")


dd = d1*d2

X3 = np.zeros((Ntrain_3,dd))
X8 = np.zeros((Ntrain_8,dd))
X9 = np.zeros((Ntrain_9,dd))

for j in range(Ntrain_3):
    img = np.squeeze(imgs_train_3[:,:,j])
    X3[j,:] = np.reshape(img,(dd,))
for j in range(Ntrain_8):
    img = np.squeeze(imgs_train_8[:,:,j])
    X8[j,:] = np.reshape(img,(dd,))
for j in range(Ntrain_9):
    img = np.squeeze(imgs_train_9[:,:,j])
    X9[j,:] = np.reshape(img,(dd,))

X = np.concatenate((X3,X8,X9),axis = 0)
print(np.shape(X))

def calculate_mean(X,NX):
    temp = copy.deepcopy(X[0,:])
    index = 1
    while index < NX:
        temp += X[index,:]
        index+=1
    return temp/NX

mean3 = calculate_mean(X3,Ntrain_3)
mean8 = calculate_mean(X8,Ntrain_8)
mean9 = calculate_mean(X9,Ntrain_9)

mean_all = calculate_mean(X,Ntrain_3+Ntrain_8+Ntrain_9)

def calculate_Si(X,NX,mean_i,dd):
    
    temp = np.zeros((dd,dd))
    index = 0
    while index < NX:
        x = X[index,:]
        vec = x - mean_i
        temp += np.outer(vec,vec)
        index+=1
    return temp


S3 = calculate_Si(X3,Ntrain_3,mean3,dd)
S8 = calculate_Si(X8,Ntrain_8,mean8,dd)
S9 = calculate_Si(X9,Ntrain_9,mean9,dd)

Sw = S3+S8+S9

Sb_3 = np.outer((mean3-mean_all),(mean3-mean_all)) *Ntrain_3
Sb_8 = np.outer((mean8-mean_all),(mean8-mean_all)) *Ntrain_8
Sb_9 = np.outer((mean9-mean_all),(mean9-mean_all)) *Ntrain_9

Sb = Sb_3 + Sb_8 + Sb_9

def Calculate_W(Sb,Sw,dd):
    eigvals, eigvecs = scipy.linalg.eigh(Sb, Sw, eigvals_only=False,subset_by_index=[dd-2, dd-1])
    return eigvals, eigvecs
eigvals, eigvecs = Calculate_W(Sb,Sw,dd)

W = copy.deepcopy(eigvecs)
W[:,0] = eigvecs[:,1]
W[:,1] = eigvecs[:,0]
Y = np.matmul(X,W)

Y3 = Y[:Ntrain_3,:]
Y8 = Y[Ntrain_3:Ntrain_3+Ntrain_8,:]
Y9 = Y[Ntrain_3+Ntrain_8:,:]








fig = plt.figure()
ax = fig.add_subplot()
# plt.rcParams.update({'font.size': 16})
plt.scatter(Y3[:,0],Y3[:,1],color = 'red',label = "3")
plt.scatter(Y8[:,0],Y8[:,1],color = 'blue',label = "8")
plt.scatter(Y9[:,0],Y9[:,1],color = 'green',label = "9")
plt.legend(loc = 'upper left')
ax.set_xlabel('LDA 1')
ax.set_ylabel('LDA 2')
plt.show()

NPCA = 2
X_centered = X - np.matmul(np.ones((Ntrain_3+Ntrain_8+Ntrain_9,1)),np.transpose(np.reshape(mean_all,(-1,1))))
U,S,Vtrans = np.linalg.svd(X_centered,full_matrices = False)
V = np.transpose(Vtrans)
X3_2pca = np.matmul(X3,V[:,:NPCA])
X8_2pca = np.matmul(X8,V[:,:NPCA])
X9_2pca = np.matmul(X9,V[:,:NPCA])
fig = plt.figure()
ax = fig.add_subplot()
# plt.rcParams.update({'font.size': 16})
plt.scatter(X3_2pca[:,0],X3_2pca[:,1],color = 'red',label = "3")
plt.scatter(X8_2pca[:,0],X8_2pca[:,1],color = 'blue',label = "8")
plt.scatter(X9_2pca[:,0],X9_2pca[:,1],color = 'green',label = "9")
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
plt.legend(loc = 'upper left')
plt.show()
'''

NPCA = 20
dd = d1*d2
Xtrain,X1,X7,U,S,V = process_train_dateset(NPCA,d1,d2,Ntrain_1,Ntrain_7,imgs_train_1,imgs_train_7)
Xtest,X1test,X7test = process_test_dateset(NPCA,d1,d2,Ntest_1,Ntest_7,imgs_test_1,imgs_test_7,V)
lbl_train,lbl_test = change_label(Ntrain_1,Ntrain_7,Ntest_1,Ntest_7,Ntrain_1,Ntest_1)
#draw_data_projection(X1,X7,V,NPCA)

# Call Levenberg-Marquardt
d = NPCA
def r_and_J(w):
    return Res_and_Jac(Xtrain,lbl_train,w)
# The quadratic surface is of the form x^\top W x + v x + b 
# The total number of parameters in W,v,b is d^2 + d + 1
# The initial guess: all parameters are ones
w = np.ones((d*d + d + 1,))
iter_max = 600
tol = 1e-3

w,Niter,Loss_vals,gradnorm_vals = direct_adam(w,iter_max,1e-3)

'''





'''


NPCA = 20
dd = d1*d2
Xtrain,X1,X7,U,S,V = process_train_dateset(NPCA,d1,d2,Ntrain_1,Ntrain_7,imgs_train_1,imgs_train_7)
Xtest,X1test,X7test = process_test_dateset(NPCA,d1,d2,Ntest_1,Ntest_7,imgs_test_1,imgs_test_7,V)
lbl_train,lbl_test = change_label(Ntrain_1,Ntrain_7,Ntest_1,Ntest_7,Ntrain_1,Ntest_1)
draw_data_projection(X1,X7,V,NPCA)

'''