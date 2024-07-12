import pandas as pd
from examples import load_batch
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
from utils import utils
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':

    data = pd.DataFrame()

    for name, loader in load_batch.get_my_batch():
        saver = pd.DataFrame()
        start = time.time()
        data_name = name
        X, y = loader[0], loader[1]

        fc = ProactiveForestClassifier(alpha=0.1)#para ejecutar proactive forest------->COMENTAR EL ALGORITMO QUE NO SE VALLA A UTILIZAR
        #fc = DecisionForestClassifier()#para ejecutar random forest

        # train, test = utils.create_one(X.to_numpy(), y.to_numpy())
        # recall, roc_auc, accracy, pcd, presi = utils.validation_train(fc,train,test)

        train, test = utils.create_k(X.to_numpy(), y.to_numpy(), k=10)
        recall, roc_auc, accracy, pcd, presi = utils.cross_validation_train(fc,train,test)#------> Medida recall y roc_auc efectuando validacion cruzada con k=10

        X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)
        fc.pruning(X_test, y_test)
        
        recall, roc_auc, accracy, pcd, presi = utils.cross_validation_train(fc,train,test)
        
        end = time.time()
        duration = (end-start) / 60
        print(f'The function was executed in {duration} minutes.')
        data[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi, duration],
                                    index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD','Presicion', 'Time Mts'])
        saver[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi, duration],
                                    index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD','Presicion', 'Time Mts'])
        print('Done:', name)
        saver.T.to_csv(f"./results/{name}.csv", header=True, index=True) #batch
        saver.T.to_excel(f"./results/{name}.xlsx", header=True, index=True) #batch
        
        data.T.to_csv("./results/Results.csv", header=True, index=True) #batch
        

    data.T.to_csv("./results/Results.csv", header=True, index=True) #batch
    
