import numpy as np
import comb
import time
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import helpers as hlp

l_arr = 1
length = 100
x_line = np.linspace(-5,5,length)
grid = comb.cartesian((x_line,x_line))

def bounds(x):
	arr = np.asarray([x.min(),x.max()])
	return np.stack((arr,arr))

def hidden_funct(x,y):
	return np.exp(-(x*x+y*y))+2*np.exp(-((x-1)*(x-1)+(y-1)*(y-1)))#np.sin(0.25*np.pi*(x*x+y*y))/(x*x+y*y)*0.5+np.random.normal(0,0.2)


def tr_data_gen(x_arr,len_arr):
	rand_index = np.random.randint(0,len(x_arr),len_arr*2)
	x_point = x_arr[rand_index[:len_arr]]
	y_point = x_arr[rand_index[len_arr:]]
	outpoint = np.hstack((x_point.reshape((-1,1)),y_point.reshape((-1,1))))
	return outpoint,hidden_funct(x_point,y_point)


x,y = tr_data_gen(x_line,l_arr)
utls = hlp.UtilityFunction(kind='ucb',kappa=1,xi=1e-9)
gp = GPR()#kernel=Matern(nu=2.5))


for i in xrange(0,20):
	gp.fit(x,y)
	y_predict,sigma = gp.predict(grid,return_std=True)
#	acq_funct = utls.utility(x_line,gp,np.amax(y))
	max_point = hlp.acq_max(ac=utls.utility,gp=gp,y_max=np.amax(y),bounds=bounds(x_line))
	print max_point
	x = np.vstack((x,max_point))
	y = np.hstack((y,hidden_funct(max_point[0],max_point[1])))
print '--------------------------------------'

f, (axarr1,axarr2) = plt.subplots(1,2,sharey=True)
axarr1.hist2d(grid[:,0],grid[:,1],bins=len(x_line),weights=hidden_funct(grid[:,0],grid[:,1]))
axarr1.scatter(x[:,0],x[:,1],c='white',s=50)
axarr1.scatter(x[-1,0],x[-1,1],c='green',s=50)
axarr1.plot(x[l_arr:,0],x[l_arr:,1],color='white')
axarr2.hist2d(grid[:,0],grid[:,1],bins=len(x_line),weights=y_predict)

'''
plt.fill_between(x_line,y_predict+sigma,y_predict-sigma,alpha=0.3)
plt.plot(x_line,acq_funct)
plt.scatter(max_point,0)'''
plt.show()
#plt.savefig('outplots/test%f.png'%time.time())


