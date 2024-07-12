import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report

from examples import load_data
from utils import utils
import warnings
from proactive_forest.estimator import DecisionForestClassifier, ProactiveForestClassifier, DecisionTreeClassifier as DTC

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

X, y = load_data.load_cmc()
X_train, X_test, y_train, y_test = utils.train_test_splitt(X, y, 0.33)

features_names = X.columns

# print(features_names)

# full_tree = DecisionTreeClassifier(random_state=42)
# full_tree.fit(X_train, y_train)

# print(classification_report(y_test, full_tree.predict(X_test)))

# max_depth = full_tree.get_depth()
# print(max_depth)


# max_depth_grid_search = GridSearchCV(
#     estimator=DecisionTreeClassifier(random_state=42),
#     scoring=make_scorer(accuracy_score),
#     param_grid={'max_depth': [ max_depth for max_depth in range(1, max_depth+1)]}
# )
# max_depth_grid_search.fit(X_train, y_train)
# print(max_depth_grid_search.best_params_)
# best_max_depth_tree = max_depth_grid_search.best_estimator_
# best_max_depth = best_max_depth_tree.get_depth()

# print(best_max_depth)
# print(classification_report(y_test, best_max_depth_tree.predict(X_test)))

# ccp_alphas = full_tree.cost_complexity_pruning_path(X_train, y_train) ["ccp_alphas"]

# ccp_alpha_grid_search = GridSearchCV(
# estimator=DecisionTreeClassifier(random_state=42),
# scoring=make_scorer(accuracy_score), 
# param_grid={"ccp_alpha": [alpha for alpha in ccp_alphas]})

# ccp_alpha_grid_search.fit(X_train, y_train)

# ccp_alpha_grid_search.best_params_
# best_ccp_alpha_tree = ccp_alpha_grid_search.best_estimator_
# ccp_alpha_max_depth = best_ccp_alpha_tree.get_depth()

# print(ccp_alpha_max_depth)
# print(classification_report(y_test, best_ccp_alpha_tree.predict(X_test)))

def get_best_params(X_train, y_train):
    
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 5),
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Ejecutar la búsqueda de cuadrícula
    grid_search.fit(X_train, y_train)

    # Mejores hiperparámetros y modelo
    print("Mejores hiperparámetros:", grid_search.best_params_)
    return grid_search.best_params_
    
# h=get_best_params(X_train, y_train)
# print(h['max_depth'])