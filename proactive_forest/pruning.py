from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from proactive_forest.sets import BaggingSet
import copy


class TreePruning(ABC):
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


class ForestPruning(ABC):
    @abstractmethod
    def pruning(self, predictor, X, y):
        pass


class StaticPruning(ForestPruning):
    @abstractmethod
    def pruning(self, predictor, X, y):
        pass


class AccuracyPruning(StaticPruning):

    def __init__(self, accuracy=None):
        self.accuracy = accuracy

    def pruning(self, predictor, X, y):
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
        if not self.accuracy:
            accuracy = accuracy_score(y, predictor.predict(X))
        else:
            accuracy = self.accuracy
        initial_accuracy = accuracy

        n = 1
        while len(predictors) > limit:
            min_delta = 100
            min_delta_tree = None
            best_accuracy = None
            # print('tree', n)
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


class EROSbPruning(StaticPruning):

    def pruning(self, predictor, X, y):
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


class ForestBasedTreePruning(StaticPruning):
    """Forest pruning base on trees prunings function.

        :param predictor: <ProactiveForestClassifier> The decision forest to be pruned
        :param X: <numpy ndaray> Feature vectors
        :param y: <numpy array> Target feature
        :return: <list> Number of initial and final trees
    """

    def __init__(self, pruning_tree_type):
        self.pruning_tree_type = pruning_tree_type

    def pruning(self, predictor, X, y):
        start_nodes = 0
        end_nodes = 0
        for i in predictor._trees:
            start_nodes += len(i.nodes)
            i.prune(X, y, predictor._encoder, self.pruning_tree_type)
            end_nodes += len(i.nodes)
        return start_nodes, end_nodes


class DynamicPruning(ForestPruning):

    def __init__(self, set_generator):
        self.set_generator = set_generator

    @abstractmethod
    def pruning(self, predictor, X, y):
        pass


class WindowThresholdPruning(DynamicPruning):

    def __init__(self, set_generator, window_size=5, diversity_threshold=0.014, accuracy_threshold=0.023, ledger=None):
        super().__init__(set_generator)
        self.window_size = window_size
        self.diversity_threshold = diversity_threshold
        self.accuracy_threshold = accuracy_threshold
        self.ledger = ledger


    def pruning(self, predictor, X, y):
        """
        Trains and prune  the decision forest classifier with (X, y).

        :param X: <numpy ndarray> An array containing the feature vectors
        :param y: <numpy array> An array containing the target features
        :return: self
        """

        n_estimators = predictor._n_estimators+1
        for i in range(1, n_estimators, self.window_size):

            generator = BaggingSet(predictor._n_instances)
            ids = generator.training_ids()
            X_train = X[ids]
            y_train = y[ids]
            ids = generator.oob_ids()
            X_test = X[ids]
            y_test = y[ids]

            prev_tree_builder = copy.deepcopy(predictor._tree_builder)
            prev_trees = copy.deepcopy(predictor._trees)
            prev_accuracy = accuracy_score(
                y_train, y_pred=predictor._no_encoder_predict(X_train))
            prev_diversity = predictor.diversity_measure(
                X_train, y_train, transform=False)

            limit = i + self.window_size if i + \
                self.window_size < n_estimators else n_estimators
            for j in range(i, limit):
                new_tree = predictor._add_tree(X_train, y_train, self.set_generator)
                if self.ledger:
                    rate = j/predictor._n_estimators
                    self.ledger.update_probabilities(new_tree, rate=rate)
                    predictor._tree_builder.feature_prob = self.ledger.probabilities

            if not (self.accept_trees(predictor, X_test, y_test, prev_diversity, prev_accuracy)):
                predictor._tree_builder = prev_tree_builder
                predictor._trees = prev_trees

            generator.clear()
        return predictor

    def accept_trees(self, predictor, X, y, prev_diversity, prev_accuracy):
        diversity = predictor.diversity_measure(X, y, transform=False)
        accuracy = accuracy_score(y, predictor._no_encoder_predict(X))
        if prev_accuracy - accuracy > self.accuracy_threshold or (prev_diversity != 1 and prev_diversity - diversity > self.diversity_threshold):
            return False
        return True
