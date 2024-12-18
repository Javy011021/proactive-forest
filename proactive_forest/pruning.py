from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score


class Pruning(ABC):
    @abstractmethod
    def pruning(self):
        pass


class TreePruning(Pruning):
    @abstractmethod
    def pruning(self, predictor, X, y, encoder):
        pass


class ReduceErrorPruning(TreePruning):

    def pruning(self, predictor, X, y, encoder):
        """Reduced error pruning function.

        :param predictor: <DecisionTree> The decision tree to be pruned
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :param encoder: <LabelEncoder> Encoder used for the labels
        :return: <None>
        """

        changed_nodes = []
        accuracy_list = []
        origin_nodes = predictor.nodes.copy()
        set_accuracy = accuracy_score(
            y, encoder.inverse_transform(predictor.predict_list(X)))

        for i in range(len(origin_nodes)):
            if origin_nodes[i].__class__.__name__ != 'DecisionLeaf':
                node_list = origin_nodes.copy()
                node_list[i] = predictor._convert_to_leaf(node_list[i])
                node_list = predictor._delete_node_brachs(node_list, i)
                predictor._order_branchs(node_list)
                predictor.nodes = node_list
                predictor.last_node_id = len(node_list)

                changed_nodes.append(node_list)
                dacc = accuracy_score(
                    y, encoder.inverse_transform(predictor.predict_list(X)))
                accuracy_list.append(dacc)

        if len(accuracy_list) != 0:
            maximum = max(accuracy_list)
            max_index = accuracy_list.index(maximum)
            if set_accuracy <= maximum:
                predictor.nodes = changed_nodes[max_index]
                predictor.last_node_id = len(changed_nodes[max_index])
                predictor._order_branchs(predictor.nodes)
                predictor.reduce_prune(X, y, encoder)


class DepthPruning(TreePruning):

    def pruning(self, predictor, X, y, encoder):
        """Depth-based pruning function.

        :param predictor: <DecisionTree> The decision tree to be pruned
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :param encoder: <LabelEncoder> Encoder used for the labels
        :return: <None>
        """

        dmax = [5, 10, 15, 20, 50, 100]
        changed_nodes = []
        accuracy_list = []
        origin_nodes = predictor.nodes.copy()
        set_accuracy = accuracy_score(
            y, encoder.inverse_transform(predictor.predict_list(X)))

        for i in dmax:
            node_list = []
            for j in range(len(origin_nodes)):
                node = origin_nodes[j]
                if node.depth < i:
                    node_list.append(node)
                elif node.depth == i:
                    node_list.append(predictor._convert_to_leaf(node))
            predictor._order_branchs(node_list)
            predictor.nodes = node_list
            predictor.last_node_id = len(node_list)

            changed_nodes.append(node_list)
            dacc = accuracy_score(
                y, encoder.inverse_transform(predictor.predict_list(X)))
            accuracy_list.append(dacc)

        maximum = max(accuracy_list)
        max_index = accuracy_list.index(maximum)
        if set_accuracy <= maximum:
            predictor.nodes = changed_nodes[max_index]
            predictor.last_node_id = len(changed_nodes[max_index])
            predictor._order_branchs(predictor.nodes)


class ForestPruning(Pruning):
    @abstractmethod
    def pruning(self, predictor, X, y, accuracy=None):
        pass


class AccuracyPruning(ForestPruning):

    def pruning(self, predictor, X, y, accuracy=None):
        """Accuracy-based pruning function.

        :param predictor: <ProactiveForestClassifier> The decision forest to be pruned
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :param accuracy: <float> Accuracy of the forest
        :return: <list> Number of initial and final trees
        """

        limit = 10
        predictors = predictor._trees
        initial_len = len(predictors)
        if not accuracy:
            accuracy = accuracy_score(y, predictor.predict(X))
        initial_accuracy = accuracy

        n = 1
        while len(predictors) > limit:
            min_delta = 100
            min_delta_tree = None
            best_accuracy = None
            print('tree', n)
            for i in range(len(predictors)):
                predictor._trees = [tree for j,
                                    tree in enumerate(predictors) if j != i]
                predictions = predictor.predict(X)
                pf_accuracy = accuracy_score(y, predictions)

                delta_T = accuracy - pf_accuracy
                if delta_T < min_delta:
                    min_delta = delta_T
                    min_delta_tree = i
                    best_accuracy = pf_accuracy

            n += 1
            if initial_accuracy <= best_accuracy:
                if min_delta_tree != None:
                    predictors = [tree for j, tree in enumerate(
                        predictors) if j != min_delta_tree]
                    accuracy = best_accuracy
            else:
                break

        predictor._trees = predictors
        return initial_len, len(predictors)


class EROSbPruning(ForestPruning):

    def pruning(self, predictor, X, y, accuracy=None):
        """Version EROS pruning function.

        :param predictor: <ProactiveForestClassifier> The decision forest to be pruned
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :param accuracy: <float> Accuracy of the forest
        :return: <list> Number of initial and final trees
        """

        predictors = predictor._trees
        initial_len = len(predictors)
        accuracy_list = []

        for i in predictors:
            result = predictor._predict_on_tree(X, i)
            predictions = predictor._encoder.inverse_transform(result)
            pf_accuracy = accuracy_score(y, predictions)
            accuracy_list.append({i: pf_accuracy})

        accuracy_list.sort(key=lambda x: list(x.values())[0], reverse=True)

        trees = []
        before_pf_accuracy = 0
        n = 1
        for i in accuracy_list:
            trees.append(list(i.keys())[0])
            predictor._trees = trees
            predictions = predictor.predict(X)
            pf_accuracy = accuracy_score(y, predictions)
            if pf_accuracy < before_pf_accuracy:
                trees.pop()
            before_pf_accuracy = pf_accuracy
            n += 1

        predictor._trees = trees
        return initial_len, len(trees)
