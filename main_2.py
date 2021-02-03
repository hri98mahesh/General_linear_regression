import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math 
import csv 

def error(x,y,thetha):
    return np.sum((y - np.dot(x,thetha))**2)/(2*x.shape[0])

def gradient(x,y,thetha):
    return np.dot(np.transpose(x),y-np.dot(x,thetha))/(x.shape[0])


# Part 1 of question 2
x1 = np.random.normal(3,2,1000000)
x2 = np.random.normal(-1,2,1000000)
y = np.random.normal(0,math.sqrt(2),1000000)
x = np.concatenate((x1,x2),axis =0)
x = np.insert(np.transpose(x.reshape((2,1000000))),0,values=1,axis=1)
thetha = np.array([3,1,2])
y_temp = np.dot(x,thetha)
y = np.concatenate((y_temp,y),axis = 0)
y = np.sum(np.transpose(y.reshape((2,1000000))), axis = 1)

# shuffling
random1 = np.arange(1000000)
np.random.shuffle(random1)
x = x[random1]
y = y[random1]
# Part 2 of question 2

thetha = np.zeros(3)
new_thetha = np.zeros(3)
i = 1
error_data = []
error_data.append(error(x,y,thetha))
thetha0 = []
thetha1 = []
thetha2 = []
thetha0.append(0)
thetha1.append(0)
thetha2.append(0)
learning_rate = 0.001
batch_size = 1000000
data_size = x.shape[0]
num_batches = data_size/batch_size
check = True
while(error(x,y,thetha)>0.000001 and check):
    old_error = error(x,y,thetha)
    for j in range(0,int(num_batches)):
        X_i = x[j*batch_size:(j+1)*batch_size,:]
        Y_i = y[j*batch_size:(j+1)*batch_size]
        thetha = thetha + learning_rate*gradient(X_i,Y_i,thetha)
        error_data.append(error(x,y,thetha))
        thetha0.append(thetha[0])
        thetha1.append(thetha[1])
        thetha2.append(thetha[2])
        if batch_size == 1 and len(error_data)>10000:
            check = np.mean(error_data[len(error_data)-10000:len(error_data)-5000])- np.mean(error_data[len(error_data)-5000:len(error_data)]) >=0.000001
        if batch_size == 100 and len(error_data)>1000:
            # print(np.mean(error_data[len(error_data)-1000:len(error_data)-500])- np.mean(error_data[len(error_data)-500:len(error_data)]))
            check = np.mean(error_data[len(error_data)-1000:len(error_data)-500])- np.mean(error_data[len(error_data)-500:len(error_data)]) >=0.000001
        if batch_size == 10000 and len(error_data)>10:
            check = np.mean(error_data[len(error_data)-10:len(error_data)-5])- np.mean(error_data[len(error_data)-5:len(error_data)]) >=0.000001
        if not check:
            break
    new_error = error(x,y,thetha)
    print('After '+str(i)+'th iteration the Error is '+str(new_error))
    print('After '+str(i)+'th iteration the change in Error is '+str(new_error-old_error))
    if batch_size == 1000000:
        check = ((abs(new_error-old_error)>=0.0000001 or i<3))
    i=i+1
print("Thetha: " + str(thetha))
        
#Now 3rd part of question 2
x_1 = []
x_2 = []
y_test = []
with open('data/q2/q2test.csv','r') as file:
    csvreader = csv.reader(file)
    first = True
    for data in csvreader:
        if first:
            first = False
        else:
            x_1.append(float(data[0]))
            x_2.append(float(data[1]))
            y_test.append(float(data[2]))

x_test = np.concatenate((x_1,x_2),axis =0)
x_test = np.transpose(x_test.reshape((2,len(x_1))))
x_test = np.insert(x_test,0,values=1,axis=1)
orig_thetha = np.array([3,1,2])
print("Error with original thetha: "+str(error(x_test,y_test,orig_thetha)))
print("Error with trained thetha: "+str(error(x_test,y_test,thetha)))

#3d PLot
fig = plt.figure()
axis = plt.axes(projection='3d')
axis.set_title('3d Mesh')
axis.set_xlabel('Thetha0')
axis.set_ylabel('thetha1')
axis.set_zlabel('thetha2')
axis.view_init(45,45)
for j in range(0,len(thetha0)):
    plt.plot(thetha0[:j],thetha1[:j],thetha2[:j],linestyle='-',marker="o",color = 'r')
    plt.draw()
    plt.pause(0.2)
plt.savefig('2c_1000000')
plt.show()

