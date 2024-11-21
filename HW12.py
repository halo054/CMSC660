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

def LevenbergMarquardt(Res_and_Jac,x,ITER_MAX,TOL):
    # minimizes loss = 0.5/n sum_{j=1}^n r_j^2(x)
    # constrained minimization problem solved at each step:
    # m(p) = grad^\top p + 0.5 p^\top Bmatr p --> min
    # subject to R - ||p|| >= 0
    # rho = [loss - loss(x + p)] / [loss - m(p)]
    
    # parameters for Levengerg-Marquardt
    RMAX = 1.;
    RMIN = 1e-12;
    RHO_GOOD = 0.75 # increase R is rho > RHO_GOOD
    RHO_BAD = 0.25 # decrease R is rho < RHO_BAD
    ETA = 0.01 # reject step if rho < ETA 
    
    # initialization
    r,J = Res_and_Jac(x)
    # print(r.size())
    # print(J.size())
    n,d = np.shape(J)
    lossvals = np.zeros(ITER_MAX)
    gradnormvals = np.zeros(ITER_MAX)
    lossvals[0] = Loss(r)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r) # grad = J^\top r
    Bmatr = np.matmul(Jtrans,J) # Bmatr = J^\top J
    gradnorm = np.linalg.norm(grad)
    gradnormvals[0] = gradnorm
    R = 0.2*RMAX # initial trust region radius
    print("iter 0: loss = ",lossvals[0]," gradnorm = ",gradnorm)
    # start iterations
    iter = 1
    while gradnorm > TOL and iter < ITER_MAX:
        Bmatr = np.matmul(Jtrans,J) + (1.e-6)*np.eye(d) # B = J^\top J
        p = (-1)*np.linalg.solve(Bmatr,grad) # p = -Bmatr^{-1}grad
        norm_p = np.linalg.norm(p)
        if norm_p > R:
            # solve grad^\top p + 0.5 p^\top Bmatr p --> min
            # subject to ||p|| = R
            gap = np.abs(norm_p - R)
            iter_lam = 0
            lam_tol = 0.01*R
            lam = 1 # initial guess for lambda in the 1D constrained minimization problems
            while gap > lam_tol:
                B1 = Bmatr + lam*np.eye(d) 
                C = np.linalg.cholesky(B1) # B1 = C C^\top
                p = -scipy.linalg.solve_triangular(np.transpose(C), \
                        scipy.linalg.solve_triangular(C,grad,lower = True),lower = False)
                norm_p = np.linalg.norm(p)
                gap = np.abs(norm_p - R)
                if gap > lam_tol:
                    q = scipy.linalg.solve_triangular(C,p,lower = True)
                    norm_q = np.linalg.norm(q)
                    lamnew = lam + ((norm_p/norm_q)**2)*(norm_p-R)/R
                    if lamnew < 0:
                        lam = 0.5*lam
                    else:
                        lam = lamnew
                    iter_lam = iter_lam + 1
                    gap = np.abs(norm_p - R)
                # print("LM, iter ",iter,":", iter_lam," substeps")
        # else:
            # print("LM, iter ",iter,": steps to the model's minimum")
        # evaluate the progress
        # print("x: ",x)
        # print("p: ",p)
        xnew = x + p
        # print("size of xnew: ",xnew.size())
        rnew,Jnew = Res_and_Jac(xnew)
        # print("rnew: ",rnew.size())
        # print("Jnew: ",Jnew.size())
        lossnew = Loss(rnew)
        rho = -(lossvals[iter-1] - lossnew)/(np.sum(grad*p) + 0.5*sum(p*np.matmul(Bmatr,p)))   
        # adjust the trust region radius
        if rho < RHO_BAD:
            R = np.max(np.array([RMIN,0.25*R]))
        elif rho > RHO_GOOD:
            R = np.min(np.array([RMAX,2.0*R]))                                       
        # accept or reject the step
        if rho > ETA:
            x = xnew
            r = rnew
            J = Jnew  
            Jtrans = np.transpose(J)
            grad = np.matmul(Jtrans,r)                                       
            gradnorm = np.linalg.norm(grad)
        lossvals[iter] = lossnew
        gradnormvals[iter] = gradnorm
        print(f"LM, iter #{iter}: loss = {lossvals[iter]:.4e}, gradnorm = {gradnorm:.4e}, rho = {rho:.4e}, R = {R:.4e}")
        iter = iter + 1    
        '''          
        fig = plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.plot(gradnormvals[0:iter],label = "||grad Loss||")
        plt.xlabel("Iteration #")
        plt.ylabel("Loss function")
        plt.yscale("log")       
        '''                      
    return x,iter,lossvals[0:iter], gradnormvals[0:iter]        

def Gauss_Newton_direction(r,J):
    n,d = np.shape(J)
    I = np.eye(d)
    Jtrans = np.transpose(J)
    JT_J = np.matmul(Jtrans,J)
    JT_J = JT_J + I*1e-6
    JT_r = np.matmul(Jtrans,r)
    P = np.linalg.solve(JT_J,-JT_r) #JT * (J*P) = -JT * r --> J*P = -(J^-T)*JT * r --> J*P = -r
    return P

def Gauss_Newton(x,iter_max,tol):
    gradnormvals = []
    lossvals = []
    c = 0.1;
    rho = 0.9
    iter = 1
    gradnorm = tol+1
    r,J = r_and_J(x)
    f = Loss(r)
    lossvals.append(f)
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    gradnormvals.append(gradnorm)
    p = Gauss_Newton_direction(r,J)
    norm_p = np.linalg.norm(p)
    while (gradnorm > tol and iter < iter_max): 
        
        if( norm_p > 1):
            p = p/norm_p
        # do backtracking line search along the direction p
        a = 1 # initial step length
        r,J = r_and_J(x + a*p)
        f_temp = Loss(r)
        cpg = c*np.dot(p,grad)
    #     print("cpg = ",cpg,"f = ",f,"f_temp = ",f_temp)
        while( f_temp > f + a*cpg ): # check Wolfe's condition 1
            print("f_temp",f_temp)
            print("f + a*cpg",f + a*cpg)
            print()
            a = a*rho
            if( a < 1e-14 ):
                print("line search failed\n");
                iter = iter_max-1
                break
            r,J = r_and_J(x + a*p)
            f_temp = Loss(r)
    #         print("f_temp = ",f_temp)
        x = x + a*p
        iter = iter + 1
    #     print("iter ",iter,": dir = ",dir,", f = ",f,", ||grad f|| = ",norm_g,", step length = ",a)
        r,J = r_and_J(x)
        f = Loss(r)
        lossvals.append(f)
        Jtrans = np.transpose(J)
        grad = np.matmul(Jtrans,r)
        gradnorm = np.linalg.norm(grad)
        gradnormvals.append(gradnorm)
        p = Gauss_Newton_direction(r,J)
        norm_p = np.linalg.norm(p)
        print(f"Gauss-Newton, iter #{iter}: loss = {f:.4e}, gradnorm = {gradnorm:.4e}, rho = {rho:.4e}, Step_size = {a:.4e}")
    
    return x,iter,lossvals, gradnormvals

def ramdomize_data(Xtrain,lbl_train):
    lbl_train = np.reshape(lbl_train, (-1,1))
    train_data_lbl_pair = np.concatenate((Xtrain,lbl_train),axis = 1)
    np.random.shuffle(train_data_lbl_pair)
    new_Xtrain = train_data_lbl_pair[:,:-1]
    lbl_train = train_data_lbl_pair[:,-1]
    return new_Xtrain,lbl_train
    
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
    Jtrans = np.transpose(J)
    grad = np.matmul(Jtrans,r)
    gradnorm = np.linalg.norm(grad)
    if( gradnorm > 1):
        grad = grad/gradnorm
    x = x - step_size*grad
    return x
    
    
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
# Dsiplay the first 100 images of 1
m1 = 10
m2 = 10
fig, axs = plt.subplots(m1,m2)
for i in range(m1):
    for j in range(m2):
        img = np.squeeze(imgs_train_1[:,:,i*m2+j])
        axs[i,j].imshow(img)
        axs[i,j].axis("off")       
        
        

# Dsiplay the first 100 images of 7
m1 = 10
m2 = 10
fig, axs = plt.subplots(m1,m2)
for i in range(m1):
    for j in range(m2):
        img = np.squeeze(imgs_train_7[:,:,i*m2+j])
        axs[i,j].imshow(img)
        axs[i,j].axis("off")      
'''
# For testing NPCA
'''
NPCA_LIST = [4,6,8,10,12,14,16,18,20,22]
miss_number_list = []
for NPCA in NPCA_LIST:
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

    w,Niter,Loss_vals,gradnorm_vals = LevenbergMarquardt(r_and_J,w,iter_max,tol)
    # Apply the learned classifier to the test set
    test0 = myquadratic(Xtrain,lbl_train,w)
    hits = np.argwhere(test0 > 0)
    misses = np.argwhere(test0 < 0)
    Nhits0 = np.size(hits)
    Nmisses0 = np.size(misses)
    print("NPCA = ",NPCA)
    print(f"TRAIN SET: {Nhits0} are classified correctly, {Nmisses0} are misclassified")

    test = myquadratic(Xtest,lbl_test,w)
    hits = np.argwhere(test > 0)
    misses = np.argwhere(test < 0)
    Nhits = np.size(hits)
    Nmisses = np.size(misses)
    miss_number_list.append(Nmisses)
    print(f"TEST SET: {Nhits} are classified correctly, {Nmisses} are misclassified")
    misses = np.squeeze(misses)
    print("Misses in the test set: ",misses)
    print()
    
   
    
plt.plot(NPCA_LIST,miss_number_list)
plt.xlabel("NPCA")
plt.ylabel("Miss Number")
plt.show()
''' 
#w,Niter,Loss_vals,gradnorm_vals = Gauss_Newton(w,iter_max,tol)
#w,Loss_vals = SGD(Xtrain,lbl_train,w,200,300,0.999,step_decay = True) 
#Xtrain,lbl_train,x,batch_size,epoch,decay_rate,step_decay
# print(w)



#For Gauss-Newton
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

w,Niter,Loss_vals,gradnorm_vals = Gauss_Newton(w,iter_max,1e-3)

# Apply the learned classifier to the test set
test0 = myquadratic(Xtrain,lbl_train,w)
hits = np.argwhere(test0 > 0)
misses = np.argwhere(test0 < 0)
Nhits0 = np.size(hits)
Nmisses0 = np.size(misses)
print("NPCA = ",NPCA)
print(f"TRAIN SET: {Nhits0} are classified correctly, {Nmisses0} are misclassified")

test = myquadratic(Xtest,lbl_test,w)
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
plt.plot(Loss_vals,label = "Loss")
plt.xlabel("Iteration #")
plt.ylabel("Function values")
plt.yscale("log")

fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.plot(gradnorm_vals)
plt.xlabel("Iteration #")
plt.ylabel("||grad f||")
plt.yscale("log")    
# Plot the norm of the gradient of the loss
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

iter_max = 1000
batch_size = 100
tol = 1e-3

w,Loss_vals,gradnorm_list = SGD(Xtrain,lbl_train,w,batch_size,iter_max,0.998,step_decay = True) 
print("Decay = 0.998, step * 0.9 each 100 iteration")
print("iter_max:",iter_max,"batch_size:",batch_size)
# Apply the learned classifier to the test set
test0 = myquadratic(Xtrain,lbl_train,w)
hits = np.argwhere(test0 > 0)
misses = np.argwhere(test0 < 0)
Nhits0 = np.size(hits)
Nmisses0 = np.size(misses)
print("NPCA = ",NPCA)
print(f"TRAIN SET: {Nhits0} are classified correctly, {Nmisses0} are misclassified")

test = myquadratic(Xtest,lbl_test,w)
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
plt.plot(Loss_vals,label = "Loss")
plt.xlabel("Iteration #")
plt.ylabel("Function values")
plt.yscale("log")

fig = plt.figure()
plt.rcParams.update({'font.size': 16})
plt.plot(gradnorm_list)
plt.xlabel("Iteration #")
plt.ylabel("||grad f||")
plt.yscale("log")    
# Plot the norm of the gradient of the loss
