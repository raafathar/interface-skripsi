import streamlit as st
from streamlit_extras.switch_page_button import switch_page


def main():

    st.set_page_config(
        page_title="Prediksi Lama Rawat Inap Pasien Demam Berdarah",
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


    st.image("assets/homepage.png")
    st.header("Prediksi Lama Rawat Inap Pasien Demam Berdarah")
    st.write("Aplikasi prediksi lama rawat inap pasien demam berdarah berdasarkan hasil tes laboratorium pasien. Terdapat beberapa parameter klinis yang digunakan pada aplikasi prediksi ini antara lain umur, jenis kelamin, trombosit, hemoglobin dan jenis demam")



    col1, col2 = st.columns([3, 11])
    
    with col1:
        if st.button("Dokumentasi 📖"):
            switch_page("documentation")

    with col2:
        if st.button("Mulai prediksi :point_right:", key="start"):
            switch_page("prediction")


if __name__ == '__main__':
    main()