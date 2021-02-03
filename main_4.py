import numpy as np
import math 
import csv 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def calculate(x,y,given):
	count =0
	arr = np.zeros((0,2))
	for i in range(0,y.shape[0]):
		if y[i] ==given:
			count = count+1
			arr = np.insert(arr,arr.shape[0],x[i][:])
	arr= (np.reshape(arr,(int(arr.shape[0]/2),2)))
	mean = np.sum(arr,axis =0)/arr.shape[0]
	mean_vector = arr-mean
	co_variance = np.dot(np.transpose(arr-mean),(arr-mean))/arr.shape[0]
	y0 = np.zeros(arr.shape[0])
	y0 = y0 + given
	return arr,mean,co_variance,y0

def normalize(x):
    mean = np.sum(x,axis =0)/x.shape[0]
    mean_vector = x - mean
    variance = np.sum(mean_vector**2,axis =0)/x.shape[0]
    x = (x-mean)/np.sqrt(variance)
    return x,mean,np.sqrt(variance)

def func(x,y,z):
	return x.dot(np.linalg.inv(y).dot(np.transpose(z)))

def zfunc(x,sig0,sig1,mean0,mean1):    
	return func(x,sig1,x) - func(mean1,sig1,x) - func(x,sig1,mean1) + func(mean1,sig1,mean1) - func(x,sig0,x) + func(mean0,sig0,x) + func(x,sig0,mean0) - func(mean0,sig0,mean0)


f1 = open('data/q4/q4x.dat','r')
f2 =open('data/q4/q4y.dat','r')
x_prime = [int(x) for x in f1.read().split()]
y_prime = [y for y in f2.read().split()]

x_dat =np.array(x_prime)
x_dat = np.reshape(x_dat,(int(x_dat.shape[0]/2),2))

y_dat = np.zeros(len(y_prime))
for i in range(0,len(y_prime)):
	if y_prime[i] =='Alaska':
		y_dat[i] = 1
	else:
		y_dat[i] = 0

#part a
x_dat_orig = np.array(x_dat)
x_dat,mean_dat,stdev = normalize(x_dat)
arr0,mean0,variance0,y0 = calculate(x_dat,y_dat,0)
arr1,mean1,variance1,y1 = calculate(x_dat,y_dat,1)
print("u0 is eqal to " + str(mean0))
print("u1 is equal to "+ str(mean1))
normalized_data = np.append(arr0-mean0,arr1-mean1,axis=0)
y_data = np.append(y0,y1,axis=0)
co_variance = np.dot(np.transpose(normalized_data),normalized_data)/normalized_data.shape[0]
print("the covariance Matrix is when assumed equal is ")
print(co_variance)

#part b

arr0_orig,tmp1,tmp2,y0 = calculate(x_dat_orig,y_dat,0)
arr1_orig,tmp1,tmp2,y1 = calculate(x_dat_orig,y_dat,1)
plt.xlabel('Fresh Water Ring Diameter')
plt.ylabel('Marine Water Ring Diameter')
plt.plot(np.transpose(arr0_orig)[0],np.transpose(arr0_orig)[1],'.g')
plt.plot(np.transpose(arr1_orig)[0],np.transpose(arr1_orig)[1],'.r')
plt.savefig('Only Points')

#part c
mean0 = np.reshape(mean0,(2,1))
mean1 = np.reshape(mean1,(2,1))
u_prime = mean0 - mean1
u_prime_T = np.transpose(u_prime)
co_variance_prime = np.linalg.inv(co_variance)
u_prime_T_co_variance_prime = u_prime_T.dot(co_variance_prime)
co_variance_prime_u_prime = co_variance_prime.dot(u_prime)
m0 = np.transpose(mean0).dot(co_variance_prime.dot(mean0))
m1 = np.transpose(mean1).dot(co_variance_prime.dot(mean1))
E = m1 - m0
slope = -1*(u_prime_T_co_variance_prime[0][0]+co_variance_prime_u_prime[0][0])/(u_prime_T_co_variance_prime[0][1]+co_variance_prime_u_prime[1][0])
intercept = -1*E[0][0]/(u_prime_T_co_variance_prime[0][1]+co_variance_prime_u_prime[1][0])
print('slope = ',slope)
print('intercept = ',intercept)
z = np.linspace(50,180,1000)
new_slope = slope*(stdev[1]/stdev[0])
new_intercept = intercept*stdev[1] + mean_dat[1] - mean_dat[0] * new_slope
plt.plot(z,z*new_slope+new_intercept,'.b')
plt.savefig('Points with line')

# Part d
print("The covariance of First is i.e Sigma0")
print(variance0)
print("The covariance of First is i.e Sigma1")
print(variance1)

#part e
mean0 = np.transpose(mean0)
mean1 = np.transpose(mean1)
print(mean_dat)
print(stdev)
x_1 = np.linspace(60,180,40)
x_2 = np.linspace(300,500,40)
X_mesh,Y_mesh = np.meshgrid(x_1,x_2)
z_grid = np.zeros((40,40))
for j in range(0,40):
	for k in range(0,40):
		z_grid[j][k] = zfunc(np.array([(X_mesh[j][k]-mean_dat[0])/stdev[0] ,(Y_mesh[j][k]-mean_dat[1])/stdev[1]]),variance0,variance1,mean0,mean1)
plt.contour(X_mesh,Y_mesh,z_grid,[0],colour = 'black')
plt.savefig('Points with line and quadatic function')
plt.show()
