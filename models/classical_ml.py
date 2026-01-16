from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier

def get_ml_models():
    # Define models in pipelines with scaling where needed
    models = {
        "TabPFN": TabPFNClassifier(),
        "Logistic Regression": make_pipeline(
            # StandardScaler(), 
            LogisticRegression(max_iter=200)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF kernel)": make_pipeline(
            # StandardScaler(), 
            SVC(kernel='rbf', probability=True, random_state=42)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "k-Nearest Neighbors": make_pipeline(
            # StandardScaler(), 
            KNeighborsClassifier(n_neighbors=5)),
        "XGBoost": XGBClassifier(
        n_estimators=50,
        learning_rate=0.01,
        max_depth=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
        ),
        "lightGBM":  LGBMClassifier(
            n_estimators=50,
            learning_rate=0.01,
            max_depth=-1,
            num_leaves=11,
            # subsample=0.8,
            # colsample_bytree=0.8,
            random_state=42, 
            verbose=-1,
            objective="binary"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=50,
            learning_rate=0.01,
            depth=4,
            verbose=False,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
        # "TabNet": TabNetClassifier(),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(96,64), max_iter=400, random_state=42)
        )
    }
    return models

