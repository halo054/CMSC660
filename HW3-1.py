import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

MNIST_data = loadmat('mnist2.mat')
MNIST_data_test = MNIST_data['imgs_test']
MNIST_data_test_label = MNIST_data['labels_test']

MNIST_data_test = np.reshape(MNIST_data_test,(400,10000))
MNIST_data_test = np.transpose(MNIST_data_test)

#print(np.shape(MNIST_data_test))
#print(np.shape(MNIST_data_test_label))
u,s,vt = np.linalg.svd(MNIST_data_test)

v = np.transpose(vt)
X_axis = np.matmul(MNIST_data_test,v[:,0]) 
Y_axis = np.matmul(MNIST_data_test,v[:,1]) 

coordinates_0_x = []
coordinates_0_y = []
coordinates_1_x = []
coordinates_1_y = []

count_0 = 0
count_1 = 0
for i in range(10000):
    if MNIST_data_test_label[i] == 0:
        coordinates_0_x.append(X_axis[i])
        coordinates_0_y.append(Y_axis[i])
    elif MNIST_data_test_label[i] == 1:
        coordinates_1_x.append(X_axis[i])
        coordinates_1_y.append(Y_axis[i])

#print("Blue for 0s, Red for 1s")
plt.scatter(coordinates_0_x, coordinates_0_y,c = 'b')
plt.scatter(coordinates_1_x, coordinates_1_y,c = 'r')
plt.title("Blue for 0s                 Red for 1s")
plt.xlabel('v1')
plt.ylabel('v2')
plt.show()



def compress(k,u,s,vt): 
    u = u[:,0:k]
    s = s[0:k,0:k]
    vt = vt[0:k,:]
    A = np.matmul(u,s)
    A = np.matmul(A,vt)
    return A

I = np.identity(400)
s = s*I

#A_3 = compress(3,u,s,vt)
A_10 = compress(10,u,s,vt)
A_20 = compress(20,u,s,vt)
A_50 = compress(50,u,s,vt)

A = np.transpose(MNIST_data_test)
#A_3 = np.transpose(A_3)
A_10 = np.transpose(A_10)
A_20 = np.transpose(A_20)
A_50 = np.transpose(A_50)

A = np.reshape(A,(20,20,10000))
#A_3 = np.reshape(A_3,(20,20,10000))
A_10 = np.reshape(A_10,(20,20,10000))
A_20 = np.reshape(A_20,(20,20,10000))
A_50 = np.reshape(A_50,(20,20,10000))

#print(np.max(A))
def show_first_four_image(A,string):
    fig = plt.figure(figsize=(8, 2.5))
    plt.title(string)
    plt.axis('off') 
    
    fig.add_subplot(1, 4, 1) 
    plt.imshow(A[:,:,0],cmap='gray',vmin=0, vmax=1)
    plt.axis('off') 
    
    fig.add_subplot(1, 4, 2) 
    plt.imshow(A[:,:,1],cmap='gray',vmin=0, vmax=1)
    plt.axis('off') 
    
    fig.add_subplot(1, 4, 3) 
    plt.imshow(A[:,:,2],cmap='gray',vmin=0, vmax=1)
    plt.axis('off') 
    
    fig.add_subplot(1, 4, 4) 
    plt.imshow(A[:,:,3],cmap='gray',vmin=0, vmax=1)
    plt.axis('off') 
    
    plt.show()

#print(np.shape(A_10))
#print("Original Image:")
show_first_four_image(A,"Original Image:")
#print()
#print("K = 10:")
show_first_four_image(A_10,"K = 10:")
#print()
#print("K = 20:")
show_first_four_image(A_20,"K = 20:")
#print()
#print("K = 50:")
show_first_four_image(A_50,"K = 50:")

#show_first_four_image(A_3,"K = 3:")