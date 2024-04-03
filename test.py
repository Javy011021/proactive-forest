from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from examples.load_data import load_iris

# Cargar un conjunto de datos (por ejemplo, el conjunto de datos Iris)
X, y = load_iris()


# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir un árbol de decisión utilizando el conjunto de entrenamiento
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluar el rendimiento del árbol en el conjunto de prueba
initial_accuracy = accuracy_score(y_test, clf.predict(X_test))

def prune_tree(tree, X_val, y_val):
    # Obtiene los nodos hoja
    leaves = tree.apply(X_val)
    
    # Inicializa la mejor precisión
    best_accuracy = accuracy_score(y_val, tree.predict(X_val))
    
    # Bucle a través de cada nodo
    for node in range(tree.tree_.node_count):
        # Si el nodo es un nodo hoja, salta a la siguiente iteración
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
            continue
            
        # Temporalmente elimina el nodo y sus descendientes
        left = tree.tree_.children_left[node]
        right = tree.tree_.children_right[node]
        tree.tree_.children_left[node] = tree.tree_.children_right[node] = -1
        
        # Calcula la precisión del árbol podado
        pruned_accuracy = accuracy_score(y_val, tree.predict(X_val))
        
        # Si la precisión mejora o permanece dentro de un umbral predefinido, acepta la poda
        if pruned_accuracy >= best_accuracy:
            best_accuracy = pruned_accuracy
        else:
            # Restaura el nodo y sus descendientes
            tree.tree_.children_left[node] = left
            tree.tree_.children_right[node] = right
    
    return tree

# Podar el árbol
pruned_tree = prune_tree(clf, X_test, y_test)

# Evaluar el rendimiento del árbol podado en el conjunto de prueba
pruned_accuracy = accuracy_score(y_test, pruned_tree.predict(X_test))
print("Precisión inicial del árbol:", initial_accuracy)
print("Precisión del árbol podado:", pruned_accuracy)
