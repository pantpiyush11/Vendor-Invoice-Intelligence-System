# from modleing_evaluation import train_random_forest, evaluate_classifier
# import joblib
# from data_preprocessing1 import load_invoice_data,apply_labels,split_data,scale_features
# from sklearn.model_selection import GridSearchCV

# FEATURES=[
#     'invoice_quantity',
#     'invoice_dollars',
#     'Freight',
#     'total_item_quantity',
#     'total_items_dollars'
# ]

# TARGET="flag_invoice"

# def main():
#     # load data
#     df=load_invoice_data()
#     df=apply_labels(df)
    
#     # Prepare Data
#     X_Train,y_Train,X_test,y_test=split_data(df,FEATURES,TARGET)
#     X_Train_Scaled,X_test_scaled=scale_features(X_Train,X_test,'Models/scaler.pkl')

#     # Train and Evaluate Models
#     grid_search=train_random_forest(X_Train_Scaled,y_Train)

#     evaluate_classifier(
#         grid_search.best_estimator_,
#         X_test_scaled,
#         y_test,
#         "Random Forest Classifier"
#     )

#     # Save best model
#     from pathlib import Path
#     model_path = Path('Models') / 'Predict_Flag_Invoice.pkl'
#     joblib.dump(grid_search.best_estimator_, model_path)

#     if __name__=="__main__":
#         main()
from modleing_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing1 import load_invoice_data, apply_labels, split_data, scale_features
import joblib
from pathlib import Path


FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_items_dollars'
]
TARGET = "flag_invoice"


def main():
    # ── Load ──────────────────────────────────────────
    df = load_invoice_data()
    df = apply_labels(df)

    # ── Prepare ───────────────────────────────────────
    # split_data returns: X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled    = scale_features(
        X_train, X_test, 'Models/scaler.pkl'
    )

    # ── Train ─────────────────────────────────────────
    grid_search = train_random_forest(X_train_scaled, y_train)

    # ── Evaluate ──────────────────────────────────────
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        model_name="Random Forest Classifier"
    )

    # ── Save ──────────────────────────────────────────
    model_path = Path('Models') / 'Predict_Flag_Invoice.pkl'
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Model saved → {model_path}")


# ✅ Outside main() — at module level
if __name__ == "__main__":
    main()