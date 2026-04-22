
# import streamlit as st
# import pandas as pd
# import numpy as np 
# import plotly.express as px

# from inference.predict_freight import predict_freight_cost
# from inference.predict_invoice_flag import predict_invoice_flag

# #--------------------------
# # Page comfiguaration
# #--------------------------
# st.set_page_config(
#     page_title="Vendor Invoice Intelligence Portal",
#     page_icon="📦",
#     layout="wide")


# #------------------------------
# #Header Section
# #------------------------------

# st.markdown("""
# # 📦Vendor Invoice Intelligence Portal
# ### AI-Driven Freight Cost Prediction and Invoice Risk Flagging System ✅
# This internal Analytics Portal leverages machine learning
# - **Forecast Freight Cost accurately.**
# - **Detect Risky and abnormal vendor invoices.**
# - **Reduce financial leakage and manual workload.**            
# """)

# st.divider()
# st.subheader("Model Selection")
# selected_model=st.sidebar.radio(
#     "Choose Prediction Module",
#     ["Freight Cost Prediction","Invoice Manual Approval Flag"] 
# )



# st.sidebar.markdown("""
# ---       
# **Business Impact**
# - 📉Improved Cost Manufacturing.
# - 🛡️Reduced Invoice Fraud & Anomalies.
# - ⚡Faster Finance Operations.                   
# """)

# if selected_model=="Freight Cost Prediction":
#     st.subheader("🚚 Freight Cost Prediction ")
#     st.markdown("""
#     **Objective:**
#     Predict Freight  Cost  for a vendor invoice using **Quantity** and **invoice Dollars** to support budgeting, forecasting, and vendor negotitations.
#     """
#           )
#     with st.form("freight_form"):
#         col1,col2=st.columns(2)
#         with col1:
#             quantity=st.number_input(
#                 "📦Quantity",
#                 min_value=1,
#                 value=1200
#             )
#         with col2:
#             dollars=st.number_input(
#                 "💰Invoice Dollars",
#                 min_value=1.0,
#                 value=18500.0 )
        
#             submit_freight=st.form_submit_button("🔮Predict Freight Cost")
        
#         if submit_freight:
#             input_data={
#             # "Quantity":[quantity],
#             "Dollars":[dollars]}
#             input_df=pd.DataFrame(input_data)
#             prediction= predict_freight_cost(input_df)
#             st.success("Prediction completed sucessfullty")
#             st.metric(
#             label="🎯 Estimated Freight Cost",
#             value=f"${prediction.loc[0,'Predicted_Freight']}"
            
#         )
# #-------------------------
# # Invoice Flagg Prediction
# #-------------------------

# else:
#     st.subheader("Invoice Manual Approval Prediction")
#     st.markdown('''
#     **Objective**
#     Predict whether a vendor invoice should be** flagged for manual approval**
#     based on abnormal cost, freight or delivery patterns.
#     ''')
#     with st.form("invoice_flag_form"):
#         col1,col2,col3=st.columns(3)
#         with col1:
#             invoice_quantity=st.number_input(
#                 "Invoice Quantity",
#                 min_value=1,
#                 value=50
#             )
#             freight=st.number_input(
#                 "freight cost",
#                 min_value=0.0,
#                 value=1.73
#             )
#         with col2:
#             invoice_dollars=st.number_input(
#                 "invoice Dollars",
#                 min_value=1.0,
#                 value=352.95                
#             )
#             total_item_quantity=st.number_input(
#                 "Total Item Quantity",
#                 min_value=1,
#                 value=162
#             )
#         with col3:
#             total_item_dollars=st.number_input(
#                 "Total item Dollars",
#                 min_value=1.0,
#                 value=2476.0
#             )
#         submit_flag=st.form_submit_button("Evaluate Invoice Risk")
#         if submit_flag:
#             input_data_2={
#                 "invoice_quantity":[invoice_quantity],
#                 "invoice_dollars":[invoice_dollars],
#                 "Freight":[freight],
#                 "total_item_quantity":[total_item_quantity],
#                 "total_item_dollars":[total_item_dollars]
#             }
#             input_data_df_2=pd.DataFrame(input_data_2)    
        
        
#             flag_prediction=predict_invoice_flag(input_data_df_2)['Predicted_Flag']
#             is_flagged=bool(flag_prediction[0])
        
#             if is_flagged:
#                 st.error("invoice requires **Manual Approval**")
#             else:
#                 st.success("invoice is safe for auto approval")





            
import streamlit as st
import pandas as pd
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ── Header ────────────────────────────────────────────
st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction and Invoice Risk Flagging ✅
This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately.**
- **Detect risky and abnormal vendor invoices.**
- **Reduce financial leakage and manual workload.**
""")
st.divider()

# ── Sidebar ───────────────────────────────────────────
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    ["Freight Cost Prediction", "Invoice Manual Approval Flag"]
)
st.sidebar.markdown("""
---
**Business Impact**
- 📉 Improved cost forecasting.
- 🛡️ Reduced invoice fraud & anomalies.
- ⚡ Faster finance operations.
""")

# ── Freight Cost Prediction ───────────────────────────
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")
    st.markdown("""
    **Objective:** Predict freight cost for a vendor invoice using
    **Quantity** and **Invoice Dollars** to support budgeting,
    forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input("📦 Quantity",       min_value=1,   value=1200)
        with col2:
            dollars  = st.number_input("💰 Invoice Dollars", min_value=1.0, value=18500.0)

        # ✅ Button outside columns, inside form
        submit_freight = st.form_submit_button("🔮 Predict Freight Cost")

    # ✅ Logic outside form block
    if submit_freight:
        input_data = {
            "Dollars":  [dollars]
        }
        prediction = predict_freight_cost(input_data)
        st.success("Prediction completed successfully!")
        st.metric(
            label="🎯 Estimated Freight Cost",
            value=f"${prediction.loc[0, 'Predicted_Freight']:,.2f}"
        )

# ── Invoice Flag Prediction ───────────────────────────
else:
    st.subheader("🚨 Invoice Manual Approval Prediction")
    st.markdown("""
    **Objective:** Predict whether a vendor invoice should be
    **flagged for manual approval** based on abnormal cost,
    freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            invoice_quantity   = st.number_input("Invoice Quantity",    min_value=1,   value=50)
            freight            = st.number_input("Freight Cost",        min_value=0.0, value=1.73)
        with col2:
            invoice_dollars    = st.number_input("Invoice Dollars",     min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1,   value=162)
        with col3:
            total_items_dollars = st.number_input("Total Item Dollars",  min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("⚠️ Evaluate Invoice Risk")

    # ✅ Logic outside form block
    if submit_flag:
        input_data = {
            "invoice_quantity":    [invoice_quantity],
            "invoice_dollars":     [invoice_dollars],
            "Freight":             [freight],
            "total_item_quantity": [total_item_quantity],
            "total_items_dollars": [total_items_dollars]   # ✅ correct key
        }

        result     = predict_invoice_flag(input_data)
        is_flagged = bool(result['Predicted_Flag'][0])
        prob       = result['Flag_Probability'][0]

        if is_flagged:
            st.error(f"⚠️ Invoice requires **Manual Approval** — Risk Score: {prob:.1%}")
        else:
            st.success(f"✅ Invoice is safe for auto-approval — Risk Score: {prob:.1%}")    
