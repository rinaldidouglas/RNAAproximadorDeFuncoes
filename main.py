import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
#Os 5 arquivos são lidos e os resultados são gerados nesta ordem, representados por "regr_1" até "regr_5" 
arquivo_1 = np.load('teste1.npy')
arquivo_2 = np.load('teste2.npy')
arquivo_3 = np.load('teste3.npy')
arquivo_4 = np.load('teste4.npy')
arquivo_5 = np.load('teste5.npy')

x_1 = arquivo_1[0]
y_1 = np.ravel(arquivo_1[1])

x_2 = arquivo_2[0]
y_2 = np.ravel(arquivo_2[1])

x_3 = arquivo_3[0]
y_3 = np.ravel(arquivo_3[1])

x_4 = arquivo_4[0]
y_4 = np.ravel(arquivo_4[1])

x_5 = arquivo_5[0]
y_5 = np.ravel(arquivo_5[1])

regrarq1 = MLPRegressor(hidden_layer_sizes = (2),
                    max_iter = 10000,
                    activation ='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver = 'adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change = 10000)
print('Treinando RNA')

regr_1 = regr_1.fit(x_1,y_1)

print('Preditor')
y_est = regr_1.predict(x_1)

plt.figure(figsize=[14, 7])

#plot curso original
plt.subplot(1, 3, 1)
plt.plot(x_1,y_1)

#plot aprendizagem
plt.subplot(1, 3, 2)
plt.plot(regr_1.loss_curve_)

#plot regressor
plt.subplot(1, 3, 3)
plt.plot(x_1,y_1,linewidth = 1,color = 'yellow')
plt.plot(x_1,y_est,linewidth = 2)

plt.show()

regr_2 = MLPRegressor(hidden_layer_sizes = (2),
                    max_iter = 10000,
                    activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver ='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change = 10000)
print('Treinando RNA')

regr_2 = regr_2.fit(x_2,y_2)

print('Preditor')
y_est = regr_2.predict(x_2)

plt.figure(figsize=[14, 7])

#plot curso original
plt.subplot(1, 3, 1)
plt.plot(x_2,y_2)

#plot aprendizagem
plt.subplot(1, 3, 2)
plt.plot(regr_2.loss_curve_)

#plot regressor
plt.subplot(1, 3, 3)
plt.plot(x_2,y_2,linewidth = 1,color = 'yellow')
plt.plot(x_2,y_est,linewidth = 2)

plt.show()

regr_3 = MLPRegressor(hidden_layer_sizes = (2),
                    max_iter = 10000,
                    activation = 'relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver = 'adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change = 10000)
print('Treinando RNA')

regr_3 = regr_3.fit(x_3,y_3)

print('Preditor')
y_est = regr_3.predict(x_3)

plt.figure(figsize = [14, 7])

#plot curso original
plt.subplot(1, 3, 1)
plt.plot(x_3,y_3)

#plot aprendizagem
plt.subplot(1, 3, 2)
plt.plot(regr_3.loss_curve_)

#plot regressor
plt.subplot(1, 3, 3)
plt.plot(x_3,y_3,linewidth = 1,color = 'yellow')
plt.plot(x_3,y_est,linewidth = 2)

plt.show()

regr_4 = MLPRegressor(hidden_layer_sizes=(2),
                    max_iter=100000,
                    activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=100000)
print('Treinando RNA')

regr_4 = regr_4.fit(x_4,y_4)

print('Preditor')
y_est = regr_4.predict(x_4)

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x_4,y_4)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr_4.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x_4,y_4,linewidth=1,color='yellow')
plt.plot(x_4,y_est,linewidth=2)

plt.show()

regr_5 = MLPRegressor(hidden_layer_sizes=(2),
                    max_iter=100000,
                    activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=100000)
print('Treinando RNA')

regr_5 = regr_5.fit(x_5,y_5)

print('Preditor')
y_est = regr_5.predict(x_5)

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x_5,y_5)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr_5.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x_5,y_5,linewidth=1,color='yellow')
plt.plot(x_5,y_est,linewidth=2)

plt.show()
