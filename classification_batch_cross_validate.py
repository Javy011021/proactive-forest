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
        data_name = name
        X, y = loader[0], loader[1]

        # para ejecutar proactive forest------->COMENTAR EL ALGORITMO QUE NO SE VALLA A UTILIZAR
        fc = ProactiveForestClassifier(alpha=0.1)
        # fc = DecisionForestClassifier()#para ejecutar random forest

        train, test = utils.create_k(X.to_numpy(), y.to_numpy(), k=10)

        start = time.time()

        recall, roc_auc, accracy, pcd, presi, incial_values, final_values = utils.cross_validation_train_with_pruning(
            fc, train, test, pruning="depth")

        end = time.time()
        duration = (end-start) / 60

        print(f'The function was executed in {duration} minutes.')
        data[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi, duration, incial_values, final_values],
                                    index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD', 'Presicion', 'Time Mts', 'Inicial_values', 'Final_values'])
        saver[data_name] = pd.Series([recall, roc_auc, accracy, pcd, presi, duration, incial_values, final_values],
                                     index=['Recall_score', 'Auc_score', 'Accuracy_score', 'Diversity_PCD', 'Presicion', 'Time Mts', 'Inicial_values', 'Final_values'])
        print('Done:', name)
        saver.T.to_csv(f"./results/{name}.csv",
                       header=True, index=True)  # batch
        saver.T.to_excel(f"./results/{name}.xlsx",
                         header=True, index=True)  # batch

        data.T.to_csv("./results/Results.csv",
                      header=True, index=True)  # batch

    data.T.to_csv("./results/Results.csv", header=True, index=True)  # batch
