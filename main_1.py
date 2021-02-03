import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def load_dataX(file):
    x = np.loadtxt("data/q1/" + file, dtype = float, skiprows=0, unpack= True)
    return x

def load_dataY(file):
    y = np.loadtxt("data/q1/" + file, dtype = float, skiprows=0, unpack= True)
    return y.transpose()

def update_thetha(thetha,x,y,learning_rate):
    error = 0
    new_thetha = np.zeros((2,1))
    m = x.shape[0]
    for j in range(0,thetha.shape[0]):
        for i in range(0,x.shape[0]):
            new_thetha[j][0] +=  (learning_rate*(y[i][0] - np.dot(x[i],thetha))*x[i][j])/x.shape[0]
        new_thetha[j][0] += thetha[j][0]
    return new_thetha

def error(x,y,thetha):
    dif = y - np.dot(x,thetha)
    return (np.sum(dif**2)/(2*x.shape[0]))

def normalize(x):
    mean = np.sum(x)/x.shape[0]
    mean_vector = x - mean
    variance = np.sum(mean_vector**2)/x.shape[0]
    x = (x-mean)/math.sqrt(variance)
    return x,mean,variance

def function(x,y,X_mesh,Y_mesh):
    Z_mesh = np.zeros((100,100))
    for i in range(0,100):
        for j in range(0,100):
            thetha = np.zeros((2,1))
            thetha[0][0] = X_mesh[i][j]
            thetha[1][0] = Y_mesh[i][j]
            Z_mesh[i][j] = error(x,y,thetha)
    return Z_mesh
    


x_orig = load_dataX("linearX.csv")
y = load_dataY("linearY.csv")
y = np.reshape(y,(y.shape[0],1))
x = np.reshape(x_orig,(x_orig.shape[0],1))
x,mean,variance = normalize(x)
x = np.insert(x,0,values=1,axis=1)
thetha = np.zeros((2,1))
new_thetha = np.zeros((2,1))
thetha_data0 = [new_thetha[0][0]]
thetha_data1 = [new_thetha[1][0]]
error_data =   [error(x,y,new_thetha)]
learning_rate =0.1
new_thetha = update_thetha(thetha,x,y,learning_rate)
i = 1
check = True
while(error(x,y,new_thetha) > 0.0000001 and check):
    thetha = new_thetha
    i = i+1
    thetha_data0.append(new_thetha[0][0])
    thetha_data1.append(new_thetha[1][0])
    error_data.append(error(x,y,new_thetha))
    new_thetha = update_thetha(thetha,x,y,learning_rate)
    check = abs(error(x,y,new_thetha) - error(x,y,thetha)) > 0.0000000001

thetha_data0.append(new_thetha[0][0])
thetha_data1.append(new_thetha[1][0])
error_data.append(error(x,y,new_thetha))
print(new_thetha)
print(error(x,y,new_thetha))

#  Question 1 part b
thetha_prime = np.zeros((2,1))
thetha_prime[1][0] = new_thetha[1][0]/math.sqrt(variance)
thetha_prime[0][0] = new_thetha[0][0]-new_thetha[1][0]*mean/(math.sqrt(variance))
plt.scatter(x_orig,y,marker='o')
z = np.linspace(4, 16, 10000)
plt.plot(z, z*thetha_prime[1][0] + thetha_prime[0][0],'-r')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Question 1 part b')
plt.savefig('1b')
plt.show()

# # # question 1 part d
# fig,ax = plt.subplots(1,1)
# xline = np.linspace(0,2, 100)
# yline = np.linspace(-1,1, 100)
# x_mesh,y_mesh = np.meshgrid(xline, yline)
# z_mesh = function(x,y,x_mesh,y_mesh)
# plt.contour(x_mesh,y_mesh,z_mesh,10)
# for i in range(0,len(error_data)):
#     plt.plot([thetha_data0[:i]],[thetha_data1[:i]],'.r')
#     plt.draw()
#     plt.pause(0.2)
# print("The loop ended PLease close the figure to play the next animation")
# plt.xlabel('thetha0')
# plt.ylabel('thetha1')
# plt.savefig('1d')
# plt.show()

# question 1 part c
fig = plt.figure()
axis = plt.axes(projection='3d')
xline = np.linspace(0,2, 100)
yline = np.linspace(-0.5,0.5, 100)
x_mesh,y_mesh = np.meshgrid(xline, yline)
z_mesh = function(x,y,x_mesh,y_mesh)
axis.plot_surface(x_mesh, y_mesh, z_mesh,cmap='viridis', edgecolor='none')
axis.view_init(60,60)
for i in range(len(error_data)):
    plt.plot([thetha_data0[i]],[thetha_data1[i]],[error_data[i]],'.r')
    plt.draw()
    plt.pause(0.2)
print("The loop ended PLease close the figure to terminate the program")
plt.savefig('1c')
plt.show()