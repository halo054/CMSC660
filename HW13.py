import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#from Levenberg_Marquardt import LevenbergMarquardt
import scipy


        
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

# Define functions for finding the optimal dividing surface
def logloss_quadratic(X,y,w):
    return 0.5*np.sum((np.log(1. + np.exp(-myquadratic(X,y,w))))**2)

def Res_and_Jac(X,y,w):
    # the vector of residuals r
    aux = np.exp(-myquadratic(X,y,w))
    r = np.log(1. + aux)
    # the Jacobian matrix J[i,j] = dr[i]/dx[j]
    a = -aux/(1. + aux)
    n,d = np.shape(X)
    d2 = d*d
    ya = y*a
    qterm = np.zeros((n,d2))
    for k in range(n):
        xk = X[k,:]
        xx = np.outer(xk,xk)
        qterm[k,:] = np.reshape(xx,(np.size(xx),))
    J = np.concatenate((qterm,X,np.ones((n,1))),axis = 1)   
    for k in range(n):
        J[k,:] = J[k,:]*ya[k]
    return r,J

def myquadratic(X,y,w):
    d = np.size(X,axis = 1)
    d2 = d*d
    W = np.reshape(w[:d2],(d,d))
    v = w[d2:d2+d];
    b = w[-1]
    qterm = np.diag(X@W@np.transpose(X))
    q = y*qterm + (np.outer(y,np.ones((d,)))*X)@v + y*b
    return q

def Loss(r):
    return 0.5*np.sum(r**2) # 0.5*sum(r^2)

def ramdomize_data(Xtrain,lbl_train):
    lbl_train = np.reshape(lbl_train, (-1,1))
    train_data_lbl_pair = np.concatenate((Xtrain,lbl_train),axis = 1)
    np.random.shuffle(train_data_lbl_pair)
    new_Xtrain = train_data_lbl_pair[:,:-1]
    lbl_train = train_data_lbl_pair[:,-1]
    return new_Xtrain,lbl_train
    
def SGD_NAG(Xtrain,lbl_train,x,batch_size,epoch,step_size):
    iter = 1
    lossvals = []
    gradnorm_list = []

    last_x = x
    while iter < epoch+1:
        new_Xtrain,new_lbl_train = ramdomize_data(Xtrain,lbl_train)
        x,last_x,loss,gradnorm = SGD_epoch_NAG(new_Xtrain,new_lbl_train,x,last_x,step_size,batch_size,iter)
        lossvals.append(loss)
        gradnorm_list.append(gradnorm)
        iter+=1
    return x , lossvals,gradnorm_list
    #return current_Xtrain,current_lbl_train
    
def SGD_epoch_NAG(Xtrain,lbl_train,x,last_x,step_size,batch_size,iteration):
    index = 0
    size = len(lbl_train)
    #batch_number = 0
    mu = 1 - 3/(5 + iteration)
    
    while index + 2*batch_size <= size:
        current_Xtrain = Xtrain[index:index + batch_size,:]
        current_lbl_train = lbl_train[index:index + batch_size]
        
        new_x = SGD_batch_NAG(current_Xtrain,current_lbl_train,x,last_x,mu,step_size)
        index += batch_size
        
        last_x = x
        x = new_x
        
        #r,J = Res_and_Jac(Xtrain,lbl_train,x)
        #loss = Loss(r)
        #print("batch_number:", batch_number,"loss:",loss)
        #batch_number+=1
    current_Xtrain = Xtrain[index:,:]
    current_lbl_train = lbl_train[index:]
    new_x = SGD_batch_NAG(current_Xtrain,current_lbl_train,x,last_x,mu,step_size)
    index += batch_size
    
    last_x = x
    x = new_x
    index += batch_size    
    
    r,J = Res_and_Jac(Xtrain,lbl_train,x)
    loss = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    print(f"NAG, iter #{iteration}: loss = {loss:.4e},  Step_size = {step_size:.4e}")
    return x,last_x,loss,gradnorm
    #print(iteration)
    #return current_Xtrain,current_lbl_train

def SGD_batch_NAG(Xtrain,lbl_train,x,last_x,mu,step_size):
    new_y = (1+mu)*x - mu*last_x
    r,J = Res_and_Jac(Xtrain,lbl_train,new_y)
    #loss = Loss(r)
    #print("loss",loss)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    if( gradnorm > 1):
        grad = grad/gradnorm

    new_x = new_y - step_size*grad
    
    return new_x
    
def SGD_Adam(Xtrain,lbl_train,x,batch_size,epoch,step_size):
    iter = 0
    lossvals = []
    gradnorm_list = []
    m = 0
    v = 0
    
    while iter < epoch:
        iter+=1
        new_Xtrain,new_lbl_train = ramdomize_data(Xtrain,lbl_train)
        x,m,v,loss,gradnorm = SGD_epoch_Adam(new_Xtrain,new_lbl_train,x,m,v,step_size,batch_size,iter)
        lossvals.append(loss)
        gradnorm_list.append(gradnorm)
        
    return x , lossvals,gradnorm_list
    #return current_Xtrain,current_lbl_train
    
def SGD_epoch_Adam(Xtrain,lbl_train,x,m,v,step_size,batch_size,iteration):
    index = 0
    size = len(lbl_train)
    batch_number = 0
    
    while index + 2*batch_size <= size:
        current_Xtrain = Xtrain[index:index + batch_size,:]
        current_lbl_train = lbl_train[index:index + batch_size]
        x,m,v = SGD_batch_Adam(current_Xtrain,current_lbl_train,x,m,v,iteration)
        index += batch_size
            

        #r,J = Res_and_Jac(Xtrain,lbl_train,x)
        #loss = Loss(r)
        #print("batch_number:", batch_number,"loss:",loss)
        batch_number+=1
    current_Xtrain = Xtrain[index:-1,:]
    current_lbl_train = lbl_train[index:-1]
    x,m,v = SGD_batch_Adam(current_Xtrain,current_lbl_train,x,m,v,iteration)
    index += batch_size    
    r,J = Res_and_Jac(Xtrain,lbl_train,x)
    loss = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    print(f"Adam, iter #{iteration}: loss = {loss:.4e},  Step_size = {step_size:.4e}")
    return x,m,v,loss,gradnorm
    #print(iteration)
    #return current_Xtrain,current_lbl_train

def SGD_batch_Adam(Xtrain,lbl_train,x,m,v,iteration,beta1 = 0.9,beta2 = 0.999, epsilon = 1e-8,step_size = 0.001):


    r,J = Res_and_Jac(Xtrain,lbl_train,x)
    loss = Loss(r)
    #print("loss",loss)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    if( gradnorm > 1):
        grad = grad/gradnorm
    
    new_m = beta1 * m + (1-beta1) * grad
    new_v = beta2 * v + (1-beta2) * grad*grad
    new_m_hat = new_m / (1-beta1**iteration)
    new_v_hat = new_v / (1-beta2**iteration)
    
    temp = new_v_hat**1/2 + epsilon
    temp = new_m_hat/temp
    
    new_x = x - step_size*temp
    
    return new_x,new_m,new_v

def direct_adam(x,iter_max,tol):
    gradnormvals = []
    lossvals = []
    iter = 1
    gradnorm = tol+1
    r,J = r_and_J(x)
    f = Loss(r)
    lossvals.append(f)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    gradnormvals.append(gradnorm)
    
    m = 0 
    v = 0 
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    step_size = 0.0001
    
    
    print(f"Adam, iter #{iter}: loss = {f:.4e}, gradnorm = {gradnorm:.4e}")
    while (gradnorm > tol and iter < iter_max): 
        if iter <=10:
            step_size = 0.0001
        else:
            step_size = 0.001
        iter+=1
        
        r,J = r_and_J(x)
        grad = np.matmul(Jtrans,r)
        gradnorm = np.linalg.norm(grad)
        print("gradnorm",gradnorm)
        if( gradnorm > 1):
            grad = grad/gradnorm
        
        new_m = beta1 * m + (1-beta1) * grad
        new_v = beta2 * v + (1-beta2) * grad*grad
        new_m_hat = new_m / (1-beta1**iter)
        new_v_hat = new_v / (1-beta2**iter)
        
        temp = new_v_hat**1/2 + epsilon
        temp = new_m_hat/temp
        
        tempnorm = np.linalg.norm(temp)
        print("tempnorm",tempnorm)
        x = x - step_size*temp
        
        r,J = r_and_J(x)
        f = Loss(r)
        lossvals.append(f)
        Jtrans = np.transpose(J)
        grad = np.matmul(Jtrans,r)
        gradnorm = np.linalg.norm(grad)
        gradnormvals.append(gradnorm)
        print(f"Adam, iter #{iter}: loss = {f:.4e}, gradnorm = {gradnorm:.4e}")
    
    return x,iter,lossvals, gradnormvals

def SGD(Xtrain,lbl_train,x,batch_size,epoch,decay_rate,step_decay = False):
    
    
    iter = 0
    lossvals = []
    gradnorm_list = []
    step_size = 1
    
    while iter < epoch:
        new_Xtrain,new_lbl_train = ramdomize_data(Xtrain,lbl_train)
        x,loss,gradnorm = SGD_epoch(new_Xtrain,new_lbl_train,x,step_size,batch_size,iter)
        lossvals.append(loss)
        gradnorm_list.append(gradnorm)
        iter+=1
        step_size = step_size * decay_rate
        if step_decay == True: 
            if iter % 100 == 0:
                step_size = step_size *0.8
                #batch_size = batch_size*2
                
    return x , lossvals,gradnorm_list
    #return current_Xtrain,current_lbl_train
    
def SGD_epoch(Xtrain,lbl_train,x,step_size,batch_size,iteration):
    index = 0
    size = len(lbl_train)
    batch_number = 0
    while index + 2*batch_size <= size:
        current_Xtrain = Xtrain[index:index + batch_size,:]
        current_lbl_train = lbl_train[index:index + batch_size]
        x = SGD_batch(current_Xtrain,current_lbl_train,x,step_size)
        index += batch_size
        #r,J = Res_and_Jac(Xtrain,lbl_train,x)
        #loss = Loss(r)
        #print("batch_number:", batch_number,"loss:",loss)
        batch_number+=1
    current_Xtrain = Xtrain[index:-1,:]
    current_lbl_train = lbl_train[index:-1]
    x = SGD_batch(current_Xtrain,current_lbl_train,x,step_size)
    r,J = Res_and_Jac(Xtrain,lbl_train,x)
    loss = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    print(f"SGD, iter #{iteration}: loss = {loss:.4e},  Step_size = {step_size:.4e}")
    return x,loss,gradnorm
    #print(iteration)
    #return current_Xtrain,current_lbl_train

def SGD_batch(Xtrain,lbl_train,x,step_size):
    r,J = Res_and_Jac(Xtrain,lbl_train,x)
    loss = Loss(r)
    #print("loss",loss)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    if( gradnorm > 1):
        grad = grad/gradnorm
    x = x - step_size*grad
    return x
    


m_list = []
v_list = []
temp_list = []
new_x_list = []










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
Itrain_1 = np.where(labels_train == 1)
Itrain_7 = np.where(labels_train == 7)
Itest_1 = np.where(labels_test == 1)
Itest_7 = np.where(labels_test == 7)
imgs_train_1 = np.squeeze(imgs_train[:,:,Itrain_1])
imgs_train_7 = np.squeeze(imgs_train[:,:,Itrain_7])
imgs_test_1 = np.squeeze(imgs_test[:,:,Itest_1])
imgs_test_7 = np.squeeze(imgs_test[:,:,Itest_7])
Ntrain_1 = np.size(Itrain_1)
Ntrain_7 = np.size(Itrain_7)
Ntest_1 = np.size(Itest_1)
Ntest_7 = np.size(Itest_7)
print(f"Ntrain_1 = {Ntrain_1}, Ntrain_7 = {Ntrain_7}")
print(f"Ntest_1 = {Ntest_1}, Ntest_7 = {Ntest_7}")



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













NPCA = 20
dd = d1*d2
Xtrain,X1,X7,U,S,V = process_train_dateset(NPCA,d1,d2,Ntrain_1,Ntrain_7,imgs_train_1,imgs_train_7)
Xtest,X1test,X7test = process_test_dateset(NPCA,d1,d2,Ntest_1,Ntest_7,imgs_test_1,imgs_test_7,V)
lbl_train,lbl_test = change_label(Ntrain_1,Ntrain_7,Ntest_1,Ntest_7,Ntrain_1,Ntest_1)
#draw_data_projection(X1,X7,V,NPCA)


d = NPCA
def r_and_J(w):
    return Res_and_Jac(Xtrain,lbl_train,w)
# The quadratic surface is of the form x^\top W x + v x + b 
# The total number of parameters in W,v,b is d^2 + d + 1
# The initial guess: all parameters are ones
w = np.ones((d*d + d + 1,))

iter_max = 600
#batch_size = len(lbl_train)
deterministic_size = len(lbl_train)
batch_size = 200
print(batch_size)
tol = 1e-3

NAG_deterministic_step_size = 0.1
NAG_stochastic_step_size = 0.005
Adam_step_size = 1
#w,Loss_vals,gradnorm_list = SGD_NAG(Xtrain,lbl_train,w,batch_size,iter_max) 


#w0,Loss_vals0,gradnorm_list0 = SGD_NAG(Xtrain,lbl_train,w,deterministic_size,iter_max,step_size = NAG_deterministic_step_size) 
#w1,Loss_vals1,gradnorm_list1 = SGD_Adam(Xtrain,lbl_train,w,batch_size,iter_max,step_size = NAG_stochastic_step_size) 

w0,Loss_vals0,gradnorm_list0 = SGD_Adam(Xtrain,lbl_train,w,deterministic_size,iter_max,Adam_step_size) 
w1,Loss_vals1,gradnorm_list1 = SGD_Adam(Xtrain,lbl_train,w,batch_size,iter_max,Adam_step_size) 

#w0,Loss_vals0,gradnorm_list0 = SGD(Xtrain,lbl_train,w,deterministic_size,iter_max,1,step_decay = False) 
#w1,Loss_vals1,gradnorm_list1 = SGD(Xtrain,lbl_train,w,batch_size,iter_max,1,step_decay = False)
print("Deterministic:")
print("iter_max:",iter_max,"batch_size:",deterministic_size,"step_size:",1,"loss:", Loss_vals0[-1])
# Apply the learned classifier to the test set
test0 = myquadratic(Xtrain,lbl_train,w0)
hits = np.argwhere(test0 > 0)
misses = np.argwhere(test0 < 0)
Nhits0 = np.size(hits)
Nmisses0 = np.size(misses)
print(f"TRAIN SET: {Nhits0} are classified correctly, {Nmisses0} are misclassified")


test = myquadratic(Xtest,lbl_test,w0)
hits = np.argwhere(test > 0)
misses = np.argwhere(test < 0)
Nhits = np.size(hits)
Nmisses = np.size(misses)
print(f"TEST SET: {Nhits} are classified correctly, {Nmisses} are misclassified")
misses = np.squeeze(misses)
print("Misses in the test set: ",misses)


print()
print("Stochastic")
print("iter_max:",iter_max,"batch_size:",batch_size,"step_size:",1,"loss:", Loss_vals1[-1])
# Apply the learned classifier to the test set
test0 = myquadratic(Xtrain,lbl_train,w1)
hits = np.argwhere(test0 > 0)
misses = np.argwhere(test0 < 0)
Nhits0 = np.size(hits)
Nmisses0 = np.size(misses)
print(f"TRAIN SET: {Nhits0} are classified correctly, {Nmisses0} are misclassified")

test = myquadratic(Xtest,lbl_test,w1)
hits = np.argwhere(test > 0)
misses = np.argwhere(test < 0)
Nhits = np.size(hits)
Nmisses = np.size(misses)
print(f"TEST SET: {Nhits} are classified correctly, {Nmisses} are misclassified")
misses = np.squeeze(misses)
print("Misses in the test set: ",misses)
print()
# Plot the loss function 

fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.plot(Loss_vals0,label = "Loss deterministic")
plt.plot(Loss_vals1,label = "Loss stochastic")
plt.legend()
plt.xlabel("Iteration #")
plt.ylabel("Function values")
plt.yscale("log")
plt.show()

fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.plot(gradnorm_list0,label = "Gradnorm deterministic")
plt.plot(gradnorm_list1,label = "Gradnorm stochastic")
plt.legend()
plt.xlabel("Iteration #")
plt.ylabel("||grad f||")
plt.yscale("log")    
# Plot the norm of the gradient of the loss
plt.show()