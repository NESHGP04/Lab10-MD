# Fase 5 - Visualización y discusión crítica

## Curvas de desempeño

Las curvas de accuracy y F1-macro comparan el baseline supervisado contra GMM semi-supervisado y Constrained K-means usando 5%, 10% y 20% de datos etiquetados. La tendencia principal muestra que GMM semi-supervisado obtiene el mejor desempeño en todos los porcentajes. En accuracy, alcanza promedios aproximados de 0.892, 0.895 y 0.901, mientras que el baseline supervisado se mantiene alrededor de 0.779, 0.826 y 0.800. La mayor mejora frente al baseline aparece con 5% de etiquetas, lo que indica que el uso de datos no etiquetados aporta más valor cuando la supervisión disponible es limitada.

El F1-macro confirma el mismo patrón. GMM semi-supervisado mantiene valores cercanos a 0.913, 0.915 y 0.919, lo que sugiere que la mejora no se concentra únicamente en las clases mayoritarias. Constrained K-means presenta resultados intermedios: puede superar al baseline en algunos escenarios, pero no alcanza la estabilidad ni el desempeño del GMM.

## Matrices de confusión

Las matrices de confusión del escenario con 10% de etiquetas permiten observar qué tan concentradas están las predicciones en la diagonal principal. El GMM semi-supervisado tiende a presentar una diagonal más marcada, señal de menos confusiones globales. El baseline supervisado comete más errores cuando las clases comparten formas morfológicas similares, porque solo aprende a partir de la fracción etiquetada. Constrained K-means depende de que las restricciones must-link y cannot-link cubran bien el espacio; cuando esa cobertura es insuficiente, puede mezclar clases cercanas aunque el agrupamiento sea relativamente coherente.

## Visualización PCA

La visualización PCA en dos dimensiones muestra que las clases reales tienen regiones parcialmente separables, pero también zonas de traslape. Ese traslape explica por qué los errores se concentran entre clases con características geométricas parecidas. Al comparar las predicciones, el GMM semi-supervisado conserva mejor la estructura general de las clases, mientras que Constrained K-means puede producir regiones más fragmentadas. Esto sugiere que el modelo generativo se beneficia de la distribución completa de los datos no etiquetados.

## Sensibilidad a hiperparámetros

La sensibilidad del GMM está influida principalmente por `covariance_type` y `reg_covar`. La configuración con covarianza `full` y mayor regularización evaluada (`reg_covar = 0.01`) obtiene el mejor desempeño promedio, cerca de 0.909 de accuracy. Las covarianzas `tied` y `spherical` son más restrictivas y reducen el desempeño, lo que apunta a underfitting al no capturar adecuadamente la geometría de cada clase.

En Constrained K-means, el número de restricciones tiene un efecto más moderado. Los promedios de accuracy para 50, 200, 500 y 1000 restricciones se mantienen cerca de 0.79-0.80. Esto indica que aumentar restricciones no garantiza una mejora proporcional; la calidad y representatividad de las restricciones puede ser más importante que la cantidad.

## Estabilidad del modelo

El boxplot entre seeds muestra que GMM semi-supervisado es el modelo más estable: combina accuracy alta con baja dispersión en los tres porcentajes de etiquetas. El baseline supervisado presenta mayor variabilidad, especialmente con 5% de etiquetas, donde la selección aleatoria de ejemplos etiquetados afecta más el entrenamiento. Constrained K-means muestra estabilidad razonable en algunos porcentajes, pero su nivel medio de desempeño es inferior al del GMM.

## Discusión crítica

Los resultados indican que los métodos semi-supervisados sí aportan valor frente al baseline supervisado, especialmente cuando la cantidad de etiquetas disponibles es baja. El aporte más fuerte se observa en GMM semi-supervisado, que utiliza las etiquetas parciales para guiar el EM y, al mismo tiempo, aprovecha la distribución de todos los puntos no etiquetados. Esto explica su mejora consistente tanto en accuracy como en F1-macro.

Constrained K-means también incorpora información supervisada mediante restricciones, pero su rendimiento depende de que esas restricciones representen bien la estructura real del dataset. En un problema con clases parcialmente solapadas, las restricciones pueden no ser suficientes para separar todas las fronteras, por lo que el modelo queda por debajo del GMM.

No se observa una señal fuerte de overfitting en GMM semi-supervisado, porque las métricas de train y test se mantienen cercanas y la variación entre seeds es baja. En cambio, el baseline muestra más inestabilidad con pocas etiquetas, lo que puede interpretarse como una limitación de aprendizaje por falta de información supervisada. En síntesis, el enfoque semi-supervisado generativo fue el más sólido para este laboratorio: funciona mejor cuando hay pocas etiquetas, mantiene buen equilibrio entre clases y ofrece una mejora clara frente al baseline.
