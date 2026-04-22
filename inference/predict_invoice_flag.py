# import joblib
# import pandas as pd

# MODEL_PATH=r"C:\Users\piyus\Desktop\Vendor Invoice Intelligence System\Models"
# def load_model(model_path: str=MODEL_PATH):
#     """
#     Load trained classfier Model.
#     """
#     with open(model_path,"rb") as f:
#         model=joblib.load(f)
#         return model

# def predict_invoice_flag(input_data):
#     """
#     Predict invoice flag for for new vendor invoices.
#     Parameters
#     input_data:dict

#     Returns
#     -------
#     pd.DataFrame with predicted flag
#     """
#     model=load_model()
#     input_df=pd.DataFrame(input_data)
#     input_df['Predicted_Flag']=model.predict(input_df).round()
#     return input_df

# if __name__=="__main__":
#     sample_data={
#         "Dollars":[18500,9000,3000,200]
#     }
#     prediction= predict_invoice_flag(sample_data)
#     print(prediction)

import joblib
import pandas as pd
from pathlib import Path

BASE        = Path(r"C:\Users\piyus\Desktop\Vendor Invoice Intelligence System\Models")
MODEL_PATH  = BASE / "Predict_Flag_Invoice.pkl"
SCALER_PATH = BASE / "scaler.pkl"

FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_items_dollars'
]

def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load trained model and scaler from disk."""
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model  loaded ← {model_path}")
    print(f"Scaler loaded ← {scaler_path}")
    return model, scaler

def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Keys must match FEATURES exactly.

    Returns
    -------
    pd.DataFrame with original data + Predicted_Flag + Flag_Probability
    """
    model, scaler = load_artifacts()

    input_df     = pd.DataFrame(input_data)[FEATURES]  # enforce column order
    input_scaled = scaler.transform(input_df)

    input_df['Predicted_Flag']    = model.predict(input_scaled)
    input_df['Flag_Probability']  = model.predict_proba(input_scaled)[:, 1].round(3)

    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity":    [10,  5,  3,  1],
        "invoice_dollars":     [18500, 9000, 3000, 200],
        "Freight":             [120,  80,  40,  10],
        "total_item_quantity": [10,   5,   3,   1],
        "total_items_dollars": [18200, 9100, 2950, 210]
    }

    predictions = predict_invoice_flag(sample_data)
    print(predictions.to_string(index=False))