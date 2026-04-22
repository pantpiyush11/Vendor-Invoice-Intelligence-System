import sqlite3
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_invoice_data():
    conn=sqlite3.connect(r'C:\Users\piyus\Desktop\Vendor Invoice Intelligence System\Data\inventory.db')
    query="""
        WITH purchase_agg AS(
    select
        p.PONumber,
        count(distinct p.Brand) as total_brands,
        sum(p.Quantity) as total_item_quantity,
        sum(p.Dollars) as total_items_dollars,
        avg(julianday(p.ReceivingDate)-julianday(p.POdate)) as avg_recieving_delay
    from purchases p
    group by p.PONumber
)
SELECT 
    vi.PONumber, 
    vi.Quantity AS invoice_quantity,
    vi.Dollars  AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate))      AS days_po_to_invoice,
    (julianday(vi.PayDate)     - julianday(vi.InvoiceDate)) AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_items_dollars,
    pa.avg_recieving_delay 
FROM vendor_invoice vi
LEFT JOIN purchase_agg pa
ON vi.PONumber =pa.PONumber
    """
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    # invoice mismatch with item level total 
    if  (abs(row["invoice_dollars"]-row["total_items_dollars"])>5):
        return 1
    # abnormally high recieveing delay
    if row["avg_recieving_delay"]>10:
        return 1
    
    return 0

def apply_labels(df):
    df['flag_invoice']=df.apply(create_invoice_risk_label,axis=1)
    return df

def split_data(df, features, target):
    X=df[features]
    y=df[target]
    return train_test_split(
    X,y, test_size=0.2, random_state=42
    )

from Pathlib import Path
def scale_features(X_train, X_test, scaler_path):
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    BASE_DIR   = Path.cwd()              # current notebook folder
    MODEL_DIR  = BASE_DIR / "models"     # subfolder
    MODEL_DIR.mkdir(exist_ok=True)       # create if doesn't exist

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled