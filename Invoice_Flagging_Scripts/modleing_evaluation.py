# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import make_scorer, classification_report,accuracy_score,f1_score
# from sklearn.model_selection import GridSearchCV


# def train_random_forest(X_train,y_train):
#     rf=RandomForestClassifier(
#         random_state=42,
#         n_jobs=-1
#     )
#     param_grid={
#         "n_estimators": [100,200,300],
#         "max_depth": [None,4,5,6],
#         "min_samples_split": [2,3,5],
#         "min_samples_leaf": [1,2,5],
#         "criterion":['gini','entropy']
#     }
#     scorer=make_scorer(f1_score)
    
#     grid_search=GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         scoring=scorer,
#         cv=5,
#         verbose=0,
#         n_jobs=-1
#     )
#     grid_search.fit(X_train,y_train)
#     return grid_search
    
# def evaluate_classifier(model,X_test,y_test,model_name):
#     preds=model.predict(X_test)
#     accuracy=accuracy_score(y_test,preds)

#     print(f"\n{model_name} Performance")
#     print(f"Accuracy:{accuracy:.2f}")
#     print(classification_report(y_test, preds
# 
#     

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, classification_report, accuracy_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1           # ✅ let GridSearchCV handle parallelism
    )
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf":  [1, 2, 5],
        "criterion":         ['gini', 'entropy']
    }
    scorer = make_scorer(f1_score, average='weighted')  # ✅ safe for any class count

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=-1          # ✅ parallelism lives here only
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest Params : {grid_search.best_params_}")
    print(f"Best CV F1  : {grid_search.best_score_:.4f}")
    return grid_search

def evaluate_classifier(model, X_test, y_test, model_name, proba=True):
    sep   = "=" * 50
    preds = model.predict(X_test)

    print(f"\n{sep}")
    print(f"  {model_name}")
    print(f"{sep}")

    # Accuracy
    print(f"  Accuracy  : {accuracy_score(y_test, preds):.4f}")

    # AUC
    if proba and hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"  ROC-AUC   : {auc:.4f}")

    # Full report
    print(f"\n{classification_report(y_test, preds)}")
    print(sep)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.show()