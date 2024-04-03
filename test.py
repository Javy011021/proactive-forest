from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar conjunto de datos
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluar la precisión en el conjunto de prueba
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Poda del árbol de decisión
# Se puede utilizar la opción ccp_alpha para controlar la intensidad de la poda
clf_postpruned = DecisionTreeClassifier(ccp_alpha=0.01)
clf_postpruned.fit(X_train, y_train)

# Evaluar la precisión en el conjunto de prueba después de la poda
y_pred_postpruned = clf_postpruned.predict(X_test)
accuracy_postpruned = accuracy_score(y_test, y_pred_postpruned)
print("Accuracy after pruning:", accuracy_postpruned)
