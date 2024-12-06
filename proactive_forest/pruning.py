from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
# from proactive_forest.tree import DecisionLeaf


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
        """Reduced error pruning function."""
        
        changedNodes = []
        accuracyList = []
        originNodes = predictor.nodes.copy()
        setAccuracy = accuracy_score(y, encoder.inverse_transform(predictor.predict_list(X)))
        
        for i in range(len(originNodes)):
            if originNodes[i].__class__.__name__ != 'DecisionLeaf':
                nodeList = originNodes.copy()
                nodeList[i] = predictor._convert_to_leaf(nodeList[i])                    
                nodeList = predictor._delete_node_brachs(nodeList, i)                
                predictor._order_branchs(nodeList)
                predictor.nodes = nodeList
                predictor.last_node_id = len(nodeList)
                
                changedNodes.append(nodeList)
                dAcc = accuracy_score(y, encoder.inverse_transform(predictor.predict_list(X)))
                accuracyList.append(dAcc)
        
        if len(accuracyList) != 0:
            maximum = max(accuracyList)
            maxindex = accuracyList.index(maximum)  
            if setAccuracy <= maximum:
                predictor.nodes = changedNodes[maxindex]
                predictor.last_node_id = len(changedNodes[maxindex])
                predictor._order_branchs(predictor.nodes)
                predictor.reduce_prune(X, y, encoder)    


class DepthPruning(TreePruning):
    
    def pruning(self, predictor, X, y, encoder):
        """Depth-based pruning function."""
        
        dmax = [5, 10, 15, 20, 50, 100]
        changedNodes = []
        accuracyList = []
        originNodes = predictor.nodes.copy()
        setAccuracy = accuracy_score(y, encoder.inverse_transform(predictor.predict_list(X)))
        
        for i in dmax:
            nodeList = [] 
            for j in range(len(originNodes)):
                node = originNodes[j]
                if node.depth < i:
                    nodeList.append(node)
                elif node.depth == i:
                    nodeList.append(predictor._convert_to_leaf(node))
            predictor._order_branchs(nodeList)   
            predictor.nodes = nodeList
            predictor.last_node_id = len(nodeList)
            
            changedNodes.append(nodeList)
            dAcc = accuracy_score(y, encoder.inverse_transform(predictor.predict_list(X)))
            accuracyList.append(dAcc)
        
        maximum = max(accuracyList)
        maxindex = accuracyList.index(maximum)  
        if setAccuracy <= maximum:
            predictor.nodes = changedNodes[maxindex]
            predictor.last_node_id = len(changedNodes[maxindex])
            predictor._order_branchs(predictor.nodes)