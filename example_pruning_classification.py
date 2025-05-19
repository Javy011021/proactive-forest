from examples import load_data
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, accuracy_score
from email_notification import send_finish_email, send_finish_file
from utils import utils
import numpy as np
import warnings
import pandas as pd
import time


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':
    tiempo_inicio = time.time()

    file_name = "./results/resultados_ejemplo_poda.csv"

    X, y = load_data.load_iris()
    print('Inicio')

    X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)

    pf = ProactiveForestClassifier(n_estimators=100, alpha=0.1, bootstrap=True)
    # pf = DecisionForestClassifier()

    pf.fit(X_train, y_train, pruning=True)  # entrenar
    pf_predictions = pf.predict(X_test)  # predecir 

    # matriz de confusion
    pf_cmat = confusion_matrix(y_test, pf_predictions)

    print("Matriz de confision: ")
    print(pf_cmat)

    pf_recall = recall_score(y_test, pf_predictions,
                             average='macro')
    print("Recall", pf_recall)

    pf_auc = utils.calculate_roc_auc(np.unique(y_train), np.unique(
        y_test), pf, X_test, y_test)  # area bajo la curva PF
    print("Area bajo la curva: ", pf_auc)

    pf_accuracy = accuracy_score(y_test, pf_predictions)  # accuracy para PF
    print("Instancias correctamente clasificadas: ", pf_accuracy)

    pf_PCD = pf.diversity_measure(X_test, y_test)
    print("Diversidad con PCD: ", pf_PCD)

    data_save = pd.DataFrame()
    data_save["Resultados PF"] = pd.Series([pf_cmat, pf_recall, pf_auc, pf_accuracy, pf_PCD, len(pf._trees)],
                                           index=['Matriz', 'Recall', 'Roc_Auc', 'Accuracy', 'Diversidad PCD', 'Final Trees'])
    data_save.T.to_csv(file_name, header=True, index=True)

    # send_finish_file(file_name);
