Classificador = MLPClassifier(hidden_layer_sizes = (15, 10), alpha = 1, max_iter = 1000)
Classificador.fit(features, target)
predicao = Classificador.predict(features)

plt.subplot(2, 2, 3)
plt.scatter(features[:, 2], features[:, 3], c = predicao, marker = 'd', cmap = 'viridis', s = 150)
plt.scatter(features[:, 2], features[:, 3], c = target, marker = 'o', cmap = 'viridis', s = 15)


plot_confusion_matrix(Classificador, features, target,include_values = True, display_labels = data.target_names)
