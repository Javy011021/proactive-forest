import pandas as pd
from examples import load_batch
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier
from utils import utils
import warnings
from sklearn.ensemble import RandomForestClassifier
import optuna

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelOptimization:

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def objective(self, trial):
        accuracy_threshold = trial.suggest_float(
            "accuracy_threshold", 0.002, 0.5)
        diversity_threshold = trial.suggest_float(
            "diversity_threshold", 0.002, 0.5)

        fc = ProactiveForestClassifier(alpha=0.1)
        # fc = DecisionForestClassifier()#para ejecutar random forest
        recall, roc_auc, accracy, pcd, presi, final_values = utils.cross_validation_train(
            fc,
            self.train,
            self.test,
            pruning=True,
            diversity_threshold=diversity_threshold,
            accuracy_threshold=accuracy_threshold)
        
        print(f'Total of trees: {len(fc._trees)}')

        return accracy* ((-1) if len(fc._trees)==100 else 1)


if __name__ == '__main__':

    # data = pd.DataFrame()

    for name, loader in load_batch.get_my_batch():
        data_name = name
        X, y = loader[0], loader[1]

        train, test = utils.create_k(X.to_numpy(), y.to_numpy(), k=2)

        model = ModelOptimization(train, test)

        study = optuna.create_study(direction="maximize")

        study.optimize(model.objective, n_trials=100)

        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value}")
        print(f"Best trial: {study.best_trial}")
        print(f"Best trials: {study.best_trials}")

    # data.T.to_csv("./results/Results.csv", header=True, index=True)
