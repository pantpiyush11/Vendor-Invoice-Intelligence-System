import joblib
from pathlib import Path

from data_preprocessing import  load_vendor_invoice_data, prepare features, split data
from model _evaluation import(
   train_linear_regression,
   train_Decision_Tree,
   train_Random_Forest,
   evaluate_model
)

def main():
    db_path="data/inventory.db"
    model_dir=Path("Models")
    model_dir.mkdir(exist_ok=True)

# Load data
df=load_vendor_invoice_data(db_path)

#prepare data
X,y=prepare_features(df)
X_train, X_test, y_train, y_test= split_data(X,y)

# Train Models
lr_model=train_linear_regression(X_train,y_train)
dt_model=train_Decision_Tree(X_train,y_train)
rf_model=train_Random_Forest(X_train,y_train)

# Evalutate models
results=[]
results.append(evaluate_model(lr_model, X_test,y_test,"Linear Regression"))
results.append(evaluate_model(dt_model, X_test,y_test,"Decision Tree Regression"))
results.append(evaluate_model(rf_model, X_test,y_test,"Random forest regression"))

# Select Best model
best_model_info =min(results, key=lambda x: x["mae"])
best_model_name =best_model_info["model_name"]

best_model={
    "Linear Regression":lr_model,
    "Decision  Tree Regression": dt_model,
    "Random forest Regression": rf_model}
[best_model_name]

# Save best model
model_path=model_dir/"predict_freight_model.pkl"
joblib.dump(best_model, model path)

print(f"\n Best model saved : { best_model_name}")
print(f"model path:{model_path}")

if __name__ = "__main__":
    main()


