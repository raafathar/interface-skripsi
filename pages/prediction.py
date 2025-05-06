import streamlit as st
import pandas as pd
import time
import joblib
from streamlit_extras.switch_page_button import switch_page


def main():

    st.set_page_config(
        page_title="Prediksi Lama Rawat Inap Pasien Demam Berdarah",
        page_icon="👩‍👦",
        initial_sidebar_state="collapsed")

    st.markdown(
        """
        <style>
            [data-testid="collapsedControl"] {
                display: none
            }
        </style>
        """, unsafe_allow_html=True,
    )

    if st.button("👈 Kembali ke Halaman Utama"):
        switch_page("app")

    st.image("assets/prediction.png")
    st.write("Masukkan beberapa isian yang digunakan untuk melakukan prediksi lama rawat inap pasien demam berdarah. Pastikan isian yang anda masukkan benar!")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2,  = st.columns(2)

    with col1:

        umur = st.text_input('Umur')
        tb = st.text_input('Trombosit')
        hct = st.text_input('Hematokrit')


    with col2:

        hb = st.text_input('Hemoglobin')
        jenis_kelamin = st.selectbox(
            "Jenis Kelamin",
            ("Laki-Laki", "Perempuan"),
            index=None,
            placeholder="Pilih jenis kelamin ... ",
        )
        jenis_demam = st.selectbox(
            "Jenis Demam",
            ("Demam Dengue", "Demam Berdarah Dengue ", "Dengue Shock Syndrome"),
            index=None,
            placeholder="Pilih jenis demam yang dialami pasien ... ",
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Hasil Prediksi"):
        if umur and tb and hb and hct and jenis_kelamin and jenis_demam:

            umur = int(umur)
            tb   = int(tb)
            hb   = float(hb)
            hct   = float(hct)

            if jenis_kelamin == "Laki-Laki":
                sex = 1
            else:
                sex = 0

            if jenis_demam == "Demam Dengue":
                jenis_demam = 0
            elif jenis_demam == "Demam Berdarah Dengue":
                jenis_demam = 1
            else:
                jenis_demam = 2

            mlp         = joblib.load('models/best_model_mlp.joblib')
            get_data    = normalized_data(sex, umur, jenis_demam, hb, hct, tb)
            result      = get_single_prediction(mlp, get_data)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander('Berdasarkan prediksi didapatkan hasil: '):
                st.success(f'Pasien akan dirawat selama {round(result[0])} hari')

            # except ValueError:
            #     time.sleep(0.5)
            #     st.toast('Teks harus berisikan angka', icon='🤧')

        else:
            time.sleep(.5)
            st.toast('Masukkan teks terlebih dahulu', icon='🤧')





def normalized_data(sex, umur, jenis_demam, hb, hct, tb):
    
    # 1. Buat DataFrame dari input
    df = pd.DataFrame({
        'umur': [umur],
        'jenis_kelamin': [sex],
        'trombosit': [tb],
        'hemoglobin': [hb],
        'hct': [hct],
        'jenis_demam': [jenis_demam]
    })

    # 2. Load MinMaxScaler
    scaler = joblib.load('models/get_minmax_scaler_model.joblib')

    # 3. Hanya ambil kolom numerik untuk transform
    numerical_features = ['umur', 'hemoglobin', 'hct', 'trombosit']

    # 4. Lakukan normalisasi
    minmax = scaler.transform(df[numerical_features])

    # 5. Masukkan hasil transform ke DataFrame
    df_scaled = pd.DataFrame(minmax, columns=numerical_features)

    # 6. Tambah kembali kolom kategorikal
    df_scaled['jenis_kelamin'] = df['jenis_kelamin']
    df_scaled['jenis_demam'] = df['jenis_demam']

    # 7. Susun ulang kolom
    model_feature_order = ['jenis_kelamin', 'umur' , 'jenis_demam', 'hemoglobin', 'hct', 'trombosit']
    df_scaled = df_scaled[model_feature_order]

    return df_scaled


def get_single_prediction(model, data):
    prediction = model.predict(data)
    return prediction

# def get_final_predict(data):

#     knn    = joblib.load('models/best_model_knn.joblib')
#     mlp     = joblib.load('models/best_model_mlp.joblib')

#     prediction_knn   = get_single_prediction(knn, data)
#     prediction_mlp   = get_single_prediction(mlp, data)

#     meta_data = pd.DataFrame({
#         'y_pred_knn' : [prediction_knn],
#         'y_pred_mlp': [prediction_mlp],
#     })

#     final_model = joblib.load('models/best_model_ensemble_svm.joblib')
#     prediction = final_model.predict(meta_data)

#     return prediction


if __name__ == '__main__':
    main()