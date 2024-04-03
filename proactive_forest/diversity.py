from abc import ABC, abstractmethod

import numpy
import numpy as np
import math
from proactive_forest.tree import DecisionLeaf, DecisionFork
from sklearn.metrics.pairwise import cosine_similarity


class DiversityMeasure(ABC):
    @abstractmethod
    def get_measure(self, predictors, X, y):
        pass


class PercentageCorrectDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Percentage of Correct Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        tally = 0
        n_instances = X.shape[0]
        for i in range(n_instances):
            instance, target = X[i], y[i]
            n_corrects = 0
            for p in predictors:
                prediction = p.predict(instance)
                if prediction == target:
                    n_corrects += 1
            if 0.1 * len(predictors) <= n_corrects <= 0.9 * len(predictors):
                tally += 1
        diversity = tally / n_instances
        return diversity


class QStatisticDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the QStatistic Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        q_total = 0
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                n = np.zeros((2, 2))
                for k in range(n_instances):
                    i_pred = predictors[i].predict(X[k])
                    j_pred = predictors[j].predict(X[k])
                    true_y = y[k]
                    if i_pred == true_y:
                        if j_pred == true_y:
                            n[1][1] += 1
                        else:
                            n[1][0] += 1
                    else:
                        if j_pred == true_y:
                            n[0][1] += 1
                        else:
                            n[0][0] += 1

                # Adding a one value to the variables which are zeros
                for k in range(2):
                    for l in range(2):
                        if n[k][l] == 0:
                            n[k][l] += 1
                same = n[1][1] * n[0][0]
                diff = n[1][0] * n[0][1]
                q_ij = (same - diff) / (same + diff)
                q_total += q_ij

        q_av = 2 * q_total / (n_predictors * (n_predictors - 1))
        return q_av


class Variance_KWDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Kohavi-Wolpert Variance Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]  # Número de instancias
        n_predictors = len(predictors)  # Número de clasificadores
        total = 0

        for i in range(n_instances):
            instance, target = X[i], y[i]
            true_positive = 0   # Número de clasificadores que clasifican correctamente la instancia
            for p in predictors:
                prediction = p.predict(instance)
                if prediction == target:
                    true_positive += 1
            total += true_positive * (n_predictors - true_positive)
        diversity = total / (n_instances * math.pow(n_predictors, 2))

        return diversity


class EntropyDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Entropy Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]  # Número de instancias
        n_predictors = len(predictors)  # Número de clasificadores
        total = 0

        for i in range(n_instances):
            instance, target = X[i], y[i]
            true_positive = 0  # Número de clasificadores que clasifican correctamente la instancia
            for p in predictors:
                prediction = p.predict(instance)
                if prediction == target:
                    true_positive += 1
            incorrect = n_predictors - true_positive
            value = min(true_positive, incorrect)
            total += value / (n_predictors - math.ceil(n_predictors/2))

        diversity = total / n_instances
        return diversity


class DoubleFaultDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Double Fault Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]  # Número de instancias
        n_predictors = len(predictors)  # Número de clasificadores

        diversity_matrix = np.zeros((n_predictors, n_predictors))  # Matriz de diversidad por pares para los árboles del bosque de decisión
        total = 0
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                matrix = np.zeros((2, 2))  # Matriz de relación en la clasificación por pares de árboles ij

                for n in range(n_instances):
                    n_target = y[n]
                    i_predict_n = predictors[i].predict(X[n])
                    j_predict_n = predictors[j].predict(X[n])

                    if n_target == i_predict_n:
                        if n_target == j_predict_n:
                            matrix[1][1] += 1
                        else:
                            matrix[1][0] += 1
                    else:
                        if n_target == j_predict_n:
                            matrix[0][1] += 1
                        else:
                            matrix[0][0] += 1

                div = matrix[0][0] / (matrix[0][1] + matrix[1][0] + matrix[0][0] + matrix[1][1]) # Se calcula la proporción de casos que han sido mal clasificados por ambos clasificadores
                diversity_matrix[i][j] = div

        for i in range(0, n_predictors - 1):
            for j in range(i + 1, n_predictors):
                total += diversity_matrix[i][j]

        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity


class DisagreementDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
        Measures the Disagreement Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]  # Número de instancias
        n_predictors = len(predictors)  # Número de clasificadores
        total = 0
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                matrix = np.zeros((2, 2))  # Matriz de relación en la clasificación por pares de árboles ij

                for n in range(n_instances):
                    n_target = y[n]
                    i_predict_n = predictors[i].predict(X[n])
                    j_predict_n = predictors[j].predict(X[n])

                    if n_target == i_predict_n:
                        if n_target == j_predict_n:
                            matrix[1][1] += 1
                        else:
                            matrix[1][0] += 1
                    else:
                        if n_target == j_predict_n:
                            matrix[0][1] += 1
                        else:
                            matrix[0][0] += 1

                dij = (matrix[0][1] + matrix[1][0]) / (matrix[0][1] + matrix[1][0] + matrix[0][0] + matrix[1][1])
                total += dij
        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity


class KagreementDiversity(DiversityMeasure):

    def get_measure(self, predictors, X, y):
        """
        Measurement of interrater agreement k

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        n_instances = X.shape[0]  # Número de instancias
        n_predictors = len(predictors)  # Número de clasificadores
        total = 0
        t_true_positive = 0  # Número total de clasificadores que clasifican correctamente cada instancias
        for i in range(n_instances):
            instance, target = X[i], y[i]
            true_positive = 0   # Número de clasificadores que clasifican correctamente la instancia
            for p in predictors:
                prediction = p.predict(instance)
                if prediction == target:
                    true_positive += 1
            kij = true_positive * (n_predictors - true_positive)
            total += kij
            t_true_positive += true_positive

        accuracy = t_true_positive / (n_instances * n_predictors)  # Precisión promedio de los clasificadores

        diversity = 1 - ((total / n_predictors) / (n_instances * (n_predictors - 1) * accuracy * (1 - accuracy)))
        return diversity


# Para cada par de árboles se determina si la característica más importante es la misma
class FeatureImportancesDiversity(DiversityMeasure):

    def get_measure(self, predictors, X, y):
        """
        Measures the Feature Importances Diversity.

        :param predictors: <list> Decision Trees
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <float>
        """
        total = 0  # Cantidad de ocasiones en las que por cada par de árboles la característica más importante es la misma
        n_predictors = len(predictors)  # Número de clasificadores
        for i in range(0, n_predictors-1):
            for j in range(i+1, n_predictors):
                i_predict_n = predictors[i]
                j_predict_n = predictors[j]
                fi_ipredictor = i_predict_n.feature_importances()
                fi_jpredictor = j_predict_n.feature_importances()
                if numpy.argmax(fi_ipredictor) == numpy.argmax(fi_jpredictor):
                    total += 1

        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity


# Para cada par de árboles se determina si para cada nivel se seleccionan las mismas características
class SelectedFeaturesDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
                Measures the Selected Features Diversity.

                :param predictors: <list> Decision Trees
                :param X: <numpy ndaray> Feature vectors
                :param y: <numpy array> Target feature
                :return: <float>
                """
        total = 0
        n_predictors = len(predictors)  # Número de clasificadores
        # arreglo tridimensional donde cada elemento representa para un árbol una matriz con la relación entre los
        # niveles y las características utilizadas en cada nivel
        tree_array = []
        tree_depth_array = np.zeros(n_predictors, int) # arreglo con la altura de cada árbol

        for i in range(0, n_predictors):
            predict = predictors[i]
            node_list = predict.nodes
            tree_depth = 0
            # Se determina la altura del árbol
            for node in node_list:
                if isinstance(node, DecisionLeaf) and node.depth >= tree_depth:
                    tree_depth = node.depth

            tree_depth_array[i] = tree_depth
            tree_array.append(np.zeros((tree_depth - 1, predict.n_features), int))  # -1 porque la raiz tiene nivel 1
            # Se determinan las características seleccionadas como criterios de división en cada nivel del árbol
            for n in node_list:
                if isinstance(n, DecisionFork):
                    feature = n.feature_id
                    depth = n.depth
                    if tree_array[i][depth - 1][feature] == 0:
                        tree_array[i][depth - 1][feature] += 1  # -1 porque la raiz tiene nivel 1

        for i in range(0, n_predictors - 1):
            features_tree_i = tree_array[i]
            depth_tree_i = tree_depth_array[i]
            for j in range(i + 1, n_predictors):
                features_tree_j = tree_array[j]
                depth_tree_j = tree_depth_array[j]
                min_depth = min(depth_tree_i, depth_tree_j)
                total_levels = depth_tree_i + depth_tree_j
                levels_count = 0
                for k in range(0, min_depth - 1):
                    if (features_tree_i[k] == features_tree_j[k]).all():
                        levels_count += 1
                total += levels_count / (total_levels - levels_count)  # índice de Jaccard

        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity


# Para cada par de árboles se determina si tienen una estructura similar.
class StructuralDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
                Measures the Structural Diversity.

                :param predictors: <list> Decision Trees
                :param X: <numpy ndaray> Feature vectors
                :param y: <numpy array> Target feature
                :return: <float>
                """
        total = 0  # Cantidad de pares de árboles con una estructura similar
        n_predictors = len(predictors)  # Número de clasificadores
        tree_depth_array = np.zeros(n_predictors, int)

        for i in range(0, n_predictors):
            predict = predictors[i]
            node_list = predict.nodes
            tree_depth = 0

            for node in node_list:
                if isinstance(node, DecisionLeaf) and node.depth >= tree_depth:
                    tree_depth = node.depth

            tree_depth_array[i] = tree_depth

        for i in range(0, n_predictors - 1):
            i_predict = predictors[i]
            for j in range(i + 1, n_predictors):
                j_predict = predictors[j]
                if i_predict.total_splits() == j_predict.total_splits() and i_predict.total_leaves() == j_predict.total_leaves() and tree_depth_array[i]==tree_depth_array[j]:
                    total += 1

        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity


# Para cada par de árboles se determina si para cada nivel la característica de mayor importancia es la misma.
class FeatureImportancesByLevelDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        """
                Measures the Feature Importances By Level Diversity.

                :param predictors: <list> Decision Trees
                :param X: <numpy ndaray> Feature vectors
                :param y: <numpy array> Target feature
                :return: <float>
                """
        total = 0
        n_predictors = len(predictors)  # Número de clasificadores
        tree_array = []
        tree_depth_array = np.zeros(n_predictors, int)

        for i in range(0, n_predictors):
            predict = predictors[i]
            node_list = predict.nodes
            tree_depth = 0

            for node in node_list:
                if isinstance(node, DecisionLeaf) and node.depth >= tree_depth:
                    tree_depth = node.depth

            tree_depth_array[i] = tree_depth
            tree_array.append(np.zeros((tree_depth - 1, predict.n_features), float))  # -1 porque la raiz tiene nivel 1

            for n in node_list:
                if isinstance(n, DecisionFork):
                    feature = n.feature_id
                    depth = n.depth
                    tree_array[i][depth - 1][feature] += n.gain * np.sum(n.samples) / np.sum(node_list[predict.root()].samples)

            for level in range(0, tree_depth-1):
                normalizer = np.sum(tree_array[i][level])
                if normalizer > 0:
                    # Avoid dividing by 0
                    tree_array[i][level] /= normalizer

        for i in range(0, n_predictors - 1):
            features_tree_i = tree_array[i]
            depth_tree_i = tree_depth_array[i]
            for j in range(i + 1, n_predictors):
                features_tree_j = tree_array[j]
                depth_tree_j = tree_depth_array[j]
                min_depth = min(depth_tree_i, depth_tree_j)
                total_levels = depth_tree_i + depth_tree_j
                levels_count = 0
                for k in range(0, min_depth - 1):
                    # Si la característica de mayor importancia es la misma en ese nivel del árbol
                    if np.argmax(features_tree_i[k]) == np.argmax(features_tree_j[k]):
                        levels_count += 1
                total += levels_count / (total_levels - levels_count)  # índice de Jaccard

        diversity = 2 * total / (n_predictors * (n_predictors - 1))
        return diversity