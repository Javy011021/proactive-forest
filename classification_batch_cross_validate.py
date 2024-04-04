import pandas as pd
from examples import load_batch
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
from utils import utils
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':

    data = pd.DataFrame()

    for name, loader in load_batch.get_my_batch():

        data_name = name
        X, y = loader[0], loader[1]

        fc = ProactiveForestClassifier(alpha=0.1)#para ejecutar proactive forest------->COMENTAR EL ALGORITMO QUE NO SE VALLA A UTILIZAR
        #fc = DecisionForestClassifier()#para ejecutar random forest

        train, test = utils.create_one(X.to_numpy(), y.to_numpy())
        recall, roc_auc, accracy, pcd, presi = utils.validation_train(fc,train,test)#

        # train, test = utils.create_k(X.to_numpy(), y.to_numpy(), k=10)
        recall, roc_auc, accracy, pcd, presi = utils.cross_validation_train(fc,train,test)#------> Medida recall y roc_auc efectuando validacion cruzada con k=10

        #print(recall, roc_auc)
        data[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi],
                                    index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD','Presicion'])
        print('Done:', name)
        data.T.to_excel("./Results.xlsx", header=True, index=True) #batch

    data.T.to_excel("./Results.xlsx", header=True, index=True) #batch
