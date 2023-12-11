from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from examples import load_batch
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
from utils import utils
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':

    data = pd.DataFrame()

    for name, loader in load_batch.get_batch_3():

        data_name = name
        X, y = loader[0], loader[1]

        fc = ProactiveForestClassifier(alpha=0.1, feature_prob=[0.18013752011991008, 0.31641150117003075, 0.009923654146116295, 0.0062430810667000805, 0.12711632459841984, 0.18150973746375662, 0.04439568498858198, 0.13426249644648436])#para ejecutar proactive forest------->COMENTAR EL ALGORITMO QUE NO SE VALLA A UTILIZAR
        #fc = DecisionForestClassifier()#para ejecutar random forest


        train, test = utils.create_k(X.to_numpy(), y.to_numpy(), k=10)
        recall, roc_auc, accracy, pcd, presi = utils.cross_validation_train(fc,train,test)#------> Medida recall y roc_auc efectuando validacion cruzada con k=10

        #print(recall, roc_auc)

        data[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi],
                                    index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD','Presicion'])
        print('Done:', name)

    data.T.to_csv("./Resultados.csv", header=True, index=True) #batch
