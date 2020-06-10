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
    X = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    X1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    Y1 = [10, 57, 63, 68, 71, 73, 75, 77, 78, 78, 80, 78, 79, 80, 81, 80]
    Y2 = [10, 56, 59, 62, 63, 63, 62, 62, 62, 61, 62, 60, 60, 61, 60, 59]
    Y3 = [2.294,1.435,1.243,1.106,1.033,0.966,0.904,0.855,0.813,0.772,0.761,0.746,0.699,0.677,0.669,0.654,0.629,0.620,0.590,0.599,0.588,0.604,0.569,0.599, 0.570,0.538,0.554,0.556,0.559,0.559]
    plt.plot(X1, Y3, label="Cross Entropy Loss")
    plt.scatter(X1, Y3, color='b')
    #plt.plot(X, Y1, label="Train Accuracy")
    #plt.scatter(X, Y1, color='b')
    #plt.plot(X, Y2, label="Test Accuracy")
    #plt.scatter(X, Y2, color='r')
    plt.xlabel("# of epochs")
    plt.ylabel("Cross Entropy Loss")
    #plt.ylabel("% Accuracy")
    plt.title("CNN(Leaky ReLU): Cross Entropy Loss\nas a function of # of epochs for SGD")
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