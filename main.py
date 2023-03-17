from sklearn import datasets
import reglog as RL
from matplotlib import pyplot as plt

data_set = datasets.load_breast_cancer()

X = data_set['data']
y = data_set['target']

LL = RL.LogicalRegression()

iteraciones, errores = LL.fit(X,y, learning_rate=0.0000000001)

plt.plot(iteraciones, errores)

plt.show()
