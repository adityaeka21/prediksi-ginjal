import streamlit as st
from web_function import predict

def app(df, x, y):
  st.title("Prediksi gagal ginjal kronis")

  col1, col2 = st.columns(2)
  with col1: 
    Bp = st.number_input("Tekanan darah", 50,180,80)
  with col2:
    Sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.030, step=0.001, format="%0.3f")

  with col1 :    
    Al = st.number_input("Albumin g/dL", 0,5,0) 
  with col2:
    Su = st.number_input("Level kadar gula pada urine",0,5,0) 

  with col1:
    Rbc = st.number_input("Jumlah sel darah merah dalam urin", 0,1,1) 
  with col2 :  
    Bu = st.number_input("Blood urea dalam urine",1.5,391.1,10.0) 

  with col1:
    Sc = st.number_input("Kadar kreatinin dalam urine mg/dL",0.4,76.0,1.2) 
  with col2:
    Sod = st.number_input("Kadar natrium dalam darah mEq/L",4.5,163.0,135.0) 

  with col1 :  
    Pot = st.number_input("Kadar pottasium dalam urine",2.5,47.0,5.0) 
  with col2:
    Hemo = st.number_input("Kadar Hemoglobin g/dL",3.1,17.8,17.0) 

  with col1:
    Wbcc = st.number_input("Jumlah sel darah putih dalam darah",2200,26400,10400) 
  with col2 :  
    Rbcc = st.number_input("Jumlah sel darah merah dalam darah", 2.1, 8.0,4.5) 
  Htn = st.selectbox("Mengidap hipertensi",["Tidak", "Ya"])

  if Htn == "Ya" : Htn = 1 
  else : Htn = 0
  features = [Bp,Sg,Al,Su,Rbc,Bu,Sc,Sod,Pot,Hemo,Wbcc,Rbcc,Htn]

  if st.button("Prediksi gagal ginjal"):
    pred, score = predict(x, y, features)
    st.info("Prediksi Sukses...")
    if(pred[0]==1):
        st.warning("Pasien mengidap gagal ginjal kronis")
    else : st.success("Pasien tidak mengidap gagal ginjal kronis")
    st.write("Model yang digunakan memiliki akurasi : ", (score*100), "%")