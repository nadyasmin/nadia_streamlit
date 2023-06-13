import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    st.title("Analisis Korelasi dan Regresi Linear Sederhana")

    # Mengunggah file
    file = st.file_uploader("Unggah file Excel", type=["xlsx", "xls"])
    if file is not None:
        df = pd.read_excel(file)

        # Menampilkan data
        st.subheader("Data")
        st.write(df)

        # Memilih kolom untuk analisis
        selected_columns = st.multiselect("Pilih kolom untuk analisis", df.columns)

        if len(selected_columns) >= 2:
            # Korelasi
            st.subheader("Korelasi")
            correlation_df = df[selected_columns].corr()
            st.write(correlation_df)

            # Regresi Linear
            st.subheader("Regresi Linear Sederhana")

            # Memilih kolom variabel untuk X dan Y
            x_column = st.selectbox("Variabel prediktor (X)", selected_columns)
            y_column = st.selectbox("Variabel respon  (Y)", selected_columns)

            # Menentukan variabel x dan y
            x = df[x_column]
            y = df[y_column]

            # Membentuk model regresi linear
            model = LinearRegression()
            model.fit(x.values.reshape(-1, 1), y)
            intercept = model.intercept_
            slope = model.coef_[0]

            # Menampilkan koefisien regresi
            st.subheader("Koefisien Regresi")
            st.write("Intercept:", intercept)
            st.write("Slope:", slope)
            
            # Menampilkan model regresi
            st.subheader("Model Regresi:")
            st.write("Y = {:.2f}X + {:.2f}".format(intercept, slope))
            
            # Mendapatkan R-square
            r_square = model.score(x.values.reshape(-1, 1), y)
            
            # Menampilkan R-Square
            st.subheader("Kebaikan Model")
            st.write("R-square: {:.2f}".format(r_square))

        else:
            st.warning("Pilih minimal 2 kolom untuk analisis.")

if __name__ == "__main__":
    main()
