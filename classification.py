from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, accuracy_score
from utils import utils
import numpy as np
import warnings
import pandas as pd


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':

    X, y = load_data.load_cmc()

    X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)

    pf = ProactiveForestClassifier(n_estimators=100,alpha=0.1, bootstrap=True)
    rf = DecisionForestClassifier()

    pf.fit(X_train, y_train)#entrenar proactive

    rf.fit(X_train, y_train)#entrenar random

    pf_predictions = pf.predict(X_test)#predecir proactive
    rf_predictions = rf.predict(X_test)#predecir random

    pf_cmat = confusion_matrix(y_test,pf_predictions)#matriz de confusion para proactive
    rf_cmat = confusion_matrix(y_test,rf_predictions)#matriz de confusion para random

    print("Matriz de confision para Proactive Forest")
    print(pf_cmat)
    print("Matriz de confision para Random Forest")
    print(rf_cmat)

    pf_recall = recall_score(y_test,pf_predictions, average = 'macro')#recall PF
    rf_recall = recall_score(y_test,rf_predictions, average = 'macro')#recall RF
    print("Recall para Proactive Forest",pf_recall)
    print("Recall para Random Forest",rf_recall)

    pf_auc = utils.calculate_roc_auc(np.unique(y_train) ,np.unique(y_test), pf, X_test, y_test)# area bajo la curva PF
    rf_auc = utils.calculate_roc_auc(np.unique(y_train) ,np.unique(y_test), rf, X_test, y_test)# area bajo la curva RF
    print("Area bajo la curva para Proactive Forest", pf_auc)
    print("Area bajo la curva para Random Forest", rf_auc)

    pf_accuracy = accuracy_score(y_test,pf_predictions)#accuracy para PF
    rf_accuracy = accuracy_score(y_test,rf_predictions)#accuracy para RF
    print("Instancias correctamente clasificadas para Proactive Forest", pf_accuracy)
    print("Instancias correctamente clasificadas para Random Forest", rf_accuracy)

    pf_PCD = pf.diversity_measure(X_test, y_test)
    rf_PCD = rf.diversity_measure(X_test, y_test)
    print("Diversidad con PCD para Proactive Forest", pf_PCD)
    print("Diversidad con PCD para Random Forest", rf_PCD)

    data_save = pd.DataFrame()
    data_save["Resultados PF"] = pd.Series([pf_cmat, pf_recall, pf_auc, pf_accuracy, pf_PCD], 
    	                                   index=['Matriz','Recall','Roc_Auc','Accuracy','Diversidad PCD'])
    data_save["Resultados RF"] = pd.Series([rf_cmat, rf_recall, rf_auc, rf_accuracy, rf_PCD], 
    	                                   index=['Matriz','Recall', 'Roc_Auc','Accuracy', 'Diversidad PCD'])
    data_save.T.to_csv("./Resultados_Un_Modelo_PF_vs_RF_.csv", header=True, index=True)
