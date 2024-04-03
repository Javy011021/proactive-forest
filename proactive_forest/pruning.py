from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import accuracy_score
from proactive_forest.estimator import ProactiveForestClassifier
from utils import utils

import random
import pprint

class Pruning(ABC):
    @abstractmethod
    def get_purnning_name():
        pass
    # def get_pruning(self, predictors, X, y):
    #     pass


class AccuracyPruning(Pruning):
    
    def get_purnning_name():
        print('Pruning by accuracy')
    
    def get_pruning(self, predictors, X, y, accuracy, limit = 50):
        """
        Measures the Percentage of Correct Diversity.

        :param predictors: <list> Decision Trees
        :param accuracy: <float> accuracy of the principal forest
        :param limit: <int> the number of trees to be left in the forest
        :return: <list> Decision Trees
        """
        while len(predictors) > limit:
            for i in range(len(predictors)):
                new_classifier = ProactiveForestClassifier(n_estimators=len(predictors) - 1)
                new_classifier._trees = [tree for j, tree in enumerate(predictors) if j != i]
                                
                X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)
                
                # new_classifier.fit(X_train, y_train)#entrenar proactive
                
                predictions = new_classifier.predict(X_test)
                pf_accuracy = accuracy_score(y_test, predictions)#accuracy para PF
                
                # Calcular Δ−T = pF - pF−T
                delta_T = accuracy - pf_accuracy
                
                # Si Δ−T es el mínimo hasta ahora, seleccionar T como el árbol menos importante para eliminar
                if delta_T < min_delta:
                    min_delta = delta_T
                    min_delta_tree = i
            
            # Eliminar el árbol menos importante
            predictors = [tree for j, tree in enumerate(predictors) if j != min_delta_tree]

        return predictions

class EROSPruning(Pruning):
    
    def get_purnning_name():
        print('Eros Pruning')
    
    def get_pruning(self, predictors, X_test, y_test):
        """
        Measures the Percentage of Correct Diversity.

        :param predictors: <list> Decision Trees
        :param accuracy: <float> accuracy of the principal forest
        :param limit: <int> the number of trees to be left in the forest
        :return: <list> Decision Trees
        """
        trees = []
        
        for i in range(len(predictors)):
            # new_classifier = ProactiveForestClassifier(n_estimators=1)
            # new_classifier._trees = [predictors[i]]
            
            # X_train, , y_train,  = utils.train_test_splitt(X, y, 0.33)
            
            # new_classifier.fit(X_train, y_train)#entrenar proactive
            
            # predictions = new_classifier.predict(X_test)
            # accuracy = accuracy_score(y_test, predictions)#accuracy para PF
            accuracy= random.random()
            trees.append({"tree": predictors[i], "accuracy": accuracy})
        
        trees_sorted = sorted(trees, key=lambda x: x["accuracy"], reverse=True)
        pprint.pprint(trees_sorted)
         
        pf_classifier = ProactiveForestClassifier(n_estimators=1)
        accuracies = []  
        
        for i in trees_sorted:
            print('ok')
            pf_classifier._n_estimators += 1
            if pf_classifier._trees:
                print('in')
                pf_classifier._trees.append(i)
            else: 
                pf_classifier._trees = [i]
                
            # X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)
            
            # new_classifier.fit(X_train, y_train)#entrenar proactive
            
            predictions = pf_classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, predictions))#accuracy para PF
        
        print(accuracies)
        
        position = pf_classifier.n_estimators
        for i in range(len(accuracies) - 3):
            if accuracies[i] > accuracies[i + 1] > accuracies[i + 2]>accuracies[i + 3]:
                position = i + 1
                break

        print(position)
        
        
        # Eliminar el árbol menores que la posicion dada
        predictors = [tree for j, tree in enumerate(predictors) if j < position]
        print(predictors)

        return predictions
