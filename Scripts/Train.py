import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_Decision_Tree,
    train_Random_Forest,
    evaluate_model
)

def main():
    db_path = r"C:\Users\piyus\Desktop\Vendor Invoice Intelligence System\Data\inventory.db"
    model_dir = Path("Models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_Decision_Tree(X_train, y_train)
    rf_model = train_Random_Forest(X_train, y_train)

    # Evaluate Models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    # Select Best Model
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]  # fix: moved bracket — was broken across lines

    # Save Best Model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()