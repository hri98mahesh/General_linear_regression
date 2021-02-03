import numpy as np
import math 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def normalize(x):
    mean = np.sum(x,axis=0)/x.shape[0]
    mean_vector = x - mean
    variance = np.sum(mean_vector**2,axis=0)/x.shape[0]
    x = np.divide((x-mean),np.sqrt(variance))
    return x,mean,np.sqrt(variance)

# def error(x,y,thetha):
#     error_value = np.multiply(y,np.log(sigmoid(x,thetha)))+np.multiply(1-y,np.log(1-sigmoid(x,thetha)))
#     return (np.sum(error_value))/

def sigmoid(x,thetha):
	return 1/(1+np.exp(-np.dot(x,thetha)))

def gradient(x,y,thetha):
	return np.dot(np.transpose(x),y-sigmoid(x,thetha))

def hasian(x,thetha):
	mat = np.zeros((3,3))
	for i in range(0,3):
		for j in range(0,3):
			mat[i][j] = np.sum(sigmoid(x,thetha)*(1-sigmoid(x,thetha))*(np.transpose(x)[i])*(np.transpose(x)[j]))
	return mat

X_data_orig = np.genfromtxt("data/q3/logisticX.csv", delimiter=",", skip_header=0)
X_data,mean,variance = normalize(X_data_orig)
X_data = np.insert(X_data,0,values=1,axis=1)
thetha = np.array([0,0,0])
new_thetha = np.array([0,0,0])
Y_data = np.genfromtxt("data/q3/logisticY.csv", delimiter=",", skip_header=0)
i =1
check = True
while(i<100 and check):
	i = i+1
	a = gradient(X_data,Y_data,thetha)
	b = np.linalg.inv(hasian(X_data,thetha))
	new_thetha = thetha +np.transpose(np.dot(np.linalg.inv(hasian(X_data,thetha)),np.transpose(gradient(X_data,Y_data,thetha))))
	check = (np.sum((new_thetha-thetha)**2)>0)
	thetha = new_thetha


for j in range(0,Y_data.shape[0]):
	if(Y_data[j]==0):
		plt.plot(X_data_orig[j][0],X_data_orig[j][1],'.r')
	else:
		plt.plot(X_data_orig[j][0],X_data_orig[j][1],'.g')
z = np.linspace(0,10,500)
new_thetha = np.zeros(3)
new_thetha[0] = thetha[0]-thetha[1]*mean[0]/variance[0]-thetha[2]*mean[1]/variance[1]
new_thetha[1] = thetha[1]/variance[0]
new_thetha[2] = thetha[2]/variance[1]
plt.plot(z,(0.5-new_thetha[0]-new_thetha[1]*z)/new_thetha[2],'.b')
plt.savefig('3_logistic')
print(thetha)
print(new_thetha)
plt.show()


