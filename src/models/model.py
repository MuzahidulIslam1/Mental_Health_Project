from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.utils.logger import get_logger

logger = get_logger("model")

def build_preprocessor():
    logger.info("Building preprocessing pipeline...")
    numeric_features = ["Age"]
    categorical_features = [
        "Gender", "family_history", "work_interfere", "no_employees",
        "remote_work", "tech_company", "benefits"
    ]

    numeric_transformer = StandardScaler()

    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    logger.info("Preprocessor ready.")
    return preprocessor


def get_models():
    logger.info("Preparing candidate model dictionary...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    logger.info(f"Models available: {list(models.keys())}")
    return models


def build_pipeline(model):
    logger.info(f"Building pipeline for {model.__class__.__name__}")
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    return pipeline
