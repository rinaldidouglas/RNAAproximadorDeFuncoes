pca = PCA(n_components = 2, whiten = True, svd_solver = 'randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_) * 100))

plt.subplot(2, 2, 2)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c = target, marker = 'o', cmap = 'viridis')

Classificador = MLPClassifier(hidden_layer_sizes = (15, 10), alpha = 1, max_iter = 1000)
Classificador.fit(features, target)

predicao = Classificador.predict(features)

plt.subplot(2, 2, 4)
plt.scatter(features[:, 2], features[:, 3], c = predicao, marker = 'd', cmap = 'viridis', s = 150)
plt.scatter(features[:, 2], features[:, 3], c = target, marker = 'o', cmap = 'viridis', s = 15)
plt.show()

plot_confusion_matrix(Classificador, features, target, include_values = True,display_labels = data.target_names)
plt.show()
