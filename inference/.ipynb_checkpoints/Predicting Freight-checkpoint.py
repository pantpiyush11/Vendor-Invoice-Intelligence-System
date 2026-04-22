import joblib
import pandas as pd

MODEL_PATH="models/predict_freight_model.pkl"
def load_model(model_path: str = MODEL_PATH):
    """
    Load Trained Freight Cost Prediction Model.
    """
    with open(model_path, "rb")as f:
        model=joblib.load(f)
        return model

def predict_freight_cost(input_data):
    """
    Predict Freight Cost for new vendors invoices.
    Parameters
    ----------
    input_dict: dict

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model=load_model()
    input_df= pd.DataFrame(input_data)
    input_df['Predicted_Freight']=model.predict(input_df).round()
    return input_df

if __name__=="__main__":
    # Example inference run local testing 
    sample_data={
        "Dollars":[18500,9000,3000,200]
    }
    prediction=prediction_freight_cost(sample_data)
    print(prediction) 