import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# sample points
X = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
Y = [10, 55, 63, 68, 70, 73, 75, 75, 77, 75, 78, 79, 79, 79, 79, 80]

# solve for a and b
def best_fit(X, Y):
    x = np.array(X)
    y = np.array(Y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m * x + c, 'r', label='Fitted line')
    plt.show()



def plot():
	epochs = 20
	X = [i for i in range(1,epochs + 1)]
	Y = [42.2300, 45.0100, 45.9400, 48., 47.2200, 48.2300, 48.2700, 48.7200, 49.0200,  49.4800, 49.8500, 49.5900, 50.0700, 50.6800, 50.5400, 51.1300, 50.2700,  50.6500,  50.6400, 50.3400]
	#plt.plot(X, Y, label="Cross Entropy Loss")
	plt.scatter(X, Y, color='b')
	#plt.plot(X, Y1, label="Train Accuracy")
	#plt.scatter(X, Y1, color='b')
	plt.plot(X, Y)
	#plt.scatter(X, Y2, color='r')
	plt.xlabel("# of epochs")
	plt.ylabel("Accuracy of MLP")
	#plt.ylabel("% Accuracy")
	plt.title("Accuracy of MLP given Epochs")
	#plt.title("CNN(Leaky ReLU): Train (blue) and Test(Red)\n accuracy as a function of # of epochs for SGD")
	plt.show()
	
def barGraph():
    objects = ('0.001', '0.01', '0.1','0.001', '0.01', '0.1','0.001', '0.01', '0.1','0.001', '0.01', '0.1')
    y_pos = np.arange(len(objects))
    print(y_pos)
    performance1 = [60.908, 49.005, 10]
    performance2 = [10, 41, 43]
    performance3 = [60.918, 48.59, 10]
    performance4 = [57.71, 50.4, 10]

    plt.bar([0,1,2], performance1, align='center', alpha=0.5, color='r', label='ReLU')
    plt.bar([3,4,5], performance2, align='center', alpha=0.5, color='b', label='Sigmoid')
    plt.bar([6,7,8], performance3, align='center', alpha=0.5, color='g', label='Leaky ReLU')
    plt.bar([9,10,11], performance4, align='center', alpha=0.5, color='y', label='Softplus')
    plt.xticks(y_pos, objects)
    plt.ylabel('% Accuracy on 5-cross validation')
    plt.legend(loc=2)
    plt.title('CNN: Hyper Parameter Tuning of the activation function \nwith different optimizer Learning Rates')

    plt.show()


plot()
#barGraph()
#best_fit(X,Y)