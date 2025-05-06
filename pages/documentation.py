import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from streamlit_extras.switch_page_button import switch_page


def main():

    st.set_page_config(
        page_title="Deteksi Penyakit Dini Anemia Pada Ibu Hamil",
        page_icon="👩‍👦",
        initial_sidebar_state="collapsed")

    # st.markdown(
    #     """
    #     <style>
    #         [data-testid="collapsedControl"] {
    #             display: none
    #         }
    #     </style>
    #     """, unsafe_allow_html=True,
    # )

    if st.button("👈 Kembali ke Halaman Utama"):
        switch_page("app")
    
    st.image("assets/documentation.png")

    st.write("Berikut merupakan beberapa dokumentasi hasil dari penelitian yang telah dilakukan untuk prediksi lama rawat inap pasien demam berdarah")
    # st.image("assets/get-started.png")

    tab1, tab2= st.tabs(
        ["(1) Ringkasan Penelitian", "(2) Implementasi Sistem"])

    with tab1:

        st.subheader("1.1 Latar Belakang")
        st.write("Demam berdarah adalah penyakit arbovirus yang ditularkan melalui gigitan nyamuk, terdapat 2 jenis nyamuk yang dapat menularkan penyakit demam berdarah antara lain Aedes Albopictus dan Aedes Aegypti. Demam berdarah telah dialami oleh jutaan orang terutama di negara tropis atau yang memiliki iklim lebih hangat (Wijayanti et al., 2025). Penderita demam berdarah ini seringkali diharuskan untuk menjalani rawat inap beberapa hari di rumah sakit. Lama rawat inap pasien demam berdarah sangat bervariatif sesuai dengan kondisi demam yang dialami, rata-rata pasien demam berdarah adalah antara 4 hari . Faktor lama rawat inap pasien demam berdarah ini dipengaruhi oleh beberapa hal seperti usia, jenis kelamin dan beberapa parameter dari hasil laboratorium (Arianti et al., 2019). Beberapa parameter dari hasil laboratorium yang dimaksud adalah trombosit, hematokrit dan hemoglobin. Secara keseluruhan ketiga parameter ini saling terkait dalam menilai faktor keadaan klinis pasien dan tingkat keparahan yang dialami. Untuk memprediksi lama rawat inap pasien demam berdarah, teknik data mining dapat digunakan untuk melakukan analisis hubungan antara variabel atau faktor yang dapat mempengaruhi lama rawat inap dari masing-masing pasien (Wiratmadja et al., 2018). Metode prediktif memungkinkan kita mengidentifikasi seberapa besar pengaruh masing-masing faktor terhadap lama tinggal di rumah sakit.")


        st.subheader("1.2 Tujuan Penelitian")
        st.write("Tujuan penelitian ini adalah untuk melakukan analisis terhadap perbandingan hasil akurasi dari Ensemble Stacking dengan metode tunggal pada studi kasus prediksi lama rawat inap pasien demam berdarah.")

        st.subheader("1.4 Alur Penelitian")
        st.write("Arsitektur sistem pada penelitian ini akan digunakan untuk proses terkait langkah-langkah yang akan digunakan selama penelitian berlangsung.")
        st.image("assets/documentation/flow_research.png", caption="Gambar 2. Metode Penelitian")

        st.subheader("1.4.1 Pengumpulan Data")
        st.write("Dataset yang digunakan pada penelitian ini didapatkan dari beberapa sumber dengan total jumlah sebanyak 848. Berikut merupakan grafik jumlah dataset yang digunakan pada penelitian ini")
        st.image("assets/documentation/chart_dataset.png", caption="Gambar 3. Grafik Dataset Penelitian")

        st.subheader("1.4.2 Transformasi Data")
        st.write("Terdapat beberapa fitur yang memiliki nilai berupa data kategorikal yaitu jenis kelamin dan jenis demam. Beberapa fitur data tersebut harus diubah nilainya atau dilakukan transformasi data ke dalam bentuk numerik. ")

        feature_transformation = pd.DataFrame(
            [
                {"Fitur": "Jenis Kelamin", "Transformasi": "P = 0 dan L = 1 "},
                {"Fitur": "Jenis Demam", "Transformasi": "DD = 0 dan DBD = 1 dan DSS = 2"},
            ]
        )

        st.dataframe(feature_transformation, use_container_width=True)


        st.subheader("1.4.2 Imputasi Missing Value KNNI")
        st.write("Data yang hilang akan dilakukan pengisian atau imputasi menggunakan salah satu metode yang terdapat pada machine learning yaitu metode KNN. Terdapat beberapa nilai missing value pada data yang digunakan, berikut merupakan jumlah nilai missing value masing-masing fitur pada grafik dibawah ini")
        st.image("assets/documentation/chart_missing_value.png", caption="Gambar 3. Jumlah Missing Value")

        st.subheader("1.4.3 Normalisasi data")
        st.write("Data akan dilakukan normalisasi sehingga memiliki rentang nilai yang sama antara 1 sampai 0 dengan menggunakan metode min-max scaler. Terdapat beberapa fitur atau kolom yang akan dinormalisasi pada penelitian ini seperti umur, trombosit, hct, dan hb. Untuk beberapa fitur atau kolom lainnya tidak lakukan normalisasi data dikarenakan data berupa numerik kategorikal seperti pada fitur jenis kelamin dan jenis demam. ")

        before_normalize = pd.DataFrame(
            [
                {"jenis_kelamin": 1, "umur": 20, "jenis_demam": 0, "hct": 36.4, "hb": 11.2, "trombosit": 142000},
                {"jenis_kelamin": 0, "umur": 40, "jenis_demam": 2, "hct": 32, "hb": 10.2, "trombosit": 60000},
                {"jenis_kelamin": 0, "umur": 50, "jenis_demam": 1, "hct": 40, "hb": 11.9, "trombosit": 110000},
            ]
        )

        st.dataframe(before_normalize, use_container_width=True)
        st.write("Berikut merupakan contoh data setelah dilakukan normalisasi")

        after_normalize = pd.DataFrame(
            [
                {"jenis_kelamin": 1, "umur": 20, "jenis_demam": 0, "hct": 0.7584, "hb": 0.6321, "trombosit": 0.4329},
                {"jenis_kelamin": 0, "umur": 40, "jenis_demam": 2, "hct": 0.5823, "hb": 0.5531, "trombosit": 0.5671},
                {"jenis_kelamin": 0, "umur": 50, "jenis_demam": 1, "hct": 0.8271, "hb": 0.3212, "trombosit": 0.8191},
            ]
        )

        st.dataframe(after_normalize, use_container_width=True)

        st.subheader("1.4.4 Pembagian data")
        st.write("Proses ini data akan dibagi menjadi dua bagian untuk melengkapi kedua proses tersebut, data yang digunakan pada saat training memiliki proporsi data yang lebih banyak jika dibandingkan pada saat testing sebagai contoh 70% untuk data testing dan 30% untuk data training. Total hasil pembagian dat dapat dilihat pada gambar 4 dibawah ini.")
        st.image("assets/documentation/chart_train_test.png", caption="Gambar 4. Pembagian Dataset")

        st.subheader("1.4.5 Hasil Akhir Skenario Pengujian")
        st.write("Proses pengujian skenario dilakukan untuk mencari metode dengan hasil terbaik baik ketika menggunakan metode tunggal dan metode pengabungan sesuai dengan skenario yang telah ditentukan. Hasil dari beberapa pengujin dapat dilihat pada tabel dibawah ini")

        record_classification = pd.DataFrame(
            [
                {"Hasil Skenario": "Skenario 1", "Accuracy":0.66, "Precission":	0.57, "Recall" :0.13, "F1-Score":	0.22},
                {"Hasil Skenario": "Skenario 2", "Accuracy":0.66, "Precission":	0.57, "Recall" :0.13, "F1-Score":	0.22},
            ]
        )

        st.dataframe(record_classification, use_container_width=True)

        st.write("Hasil dari skenario pengujian menggunakan pendekatan klasifikasi belum menemukan hasil yang maksimal, dari beberapa skenario yang dilakukan hasil terbaik diperoleh pada skenario keempat dengan hasil accuracy sebesar 0.68. Berdasarkan hal tersebut kemudian dilakukan analisis hasil secara ulang menggunakan pendekatan regresi dengan hasil akhir nanti berupa nilai prediksi lama rawat inap yang sebenarnya bukan dalam bentuk kategorikal. Hasil menggunakan pendekatan regresi dapat dilihat pada tabel dibawah ini.")

        record_regression = pd.DataFrame(
            [
                {"Hasil Skenario": "Skenario 1", "RMSE":2.78652, "MSE": 1.66929},
                {"Hasil Skenario": "Skenario 2", "RMSE":2.83957, "MSE": 1.68510},
            ]
        )

        st.dataframe(record_regression, use_container_width=True)

        st.write("Keterangan :")
        st.write("Skenario 1 : Melakukan pengujian pada masing-masing metode tunggal yaitu  Logistic Reggression, KNN, ANN dan SVM. Pengujian disertai dengan hyperparameter tuning menggunakan gridsearch")
        st.write("Skenario 2 : Melakukan pengujian pada ensemble stacking dengan mengganti meta-classifier secara bergantian yaitu Logistic Reggression, KNN, ANN dan SVM")

        st.subheader("1.5 Kesimpulan")
        st.write("1. Penggunaan metode Ensemble Stacking belum dapat memberikan peningkatan yang signifikan, bahkan hasil yang diberikan  memiliki nilai yang sama pada metode tunggal SVM, meta-classifier LR, dan meta-classifier ANN dengan nilai evaluasi yang dihasilkan untuk accuracy sebesar 0.68, precission sebesar 0.71 recall sebesar 0.17, F1-Score sebesar 0.27. Kemudian berbeda dengan hasil ketika menggunakan pendekatan regresi, pada pendekatan regresi ini justru pada saat menggunakan metode Ensemble Stacking maka didapatkan nilai error yang lebih tinggi dengan hasil terbaik pada meta-classifier SVM untuk nilai MSE sebesar 2.83957 RMSE sebesar 1.68510. Hasilnya lebih baik ketika menggunakan metode tunggal SVM dengan nilai MSE sebesar 2.78652 dan RMSE sebesar 1.66929")
        st.write("2. Hasilnya adalah metode Ensemble Stacking yang digunakan belum sepenuhnya dapat bekerja secara maksimal bahkan tidak memberikan hasil peningkatan. Hal ini dipengaruhi oleh kinerja model dasar yang kurang optimal dalam memberikan informasi yang cukup bagi meta-classifier. Jika model dasar tidak menghasilkan prediksi yang cukup berkualitas maka meta-classifier akan kesulitan untuk membuat keputusan yang lebih baik.")



        st.subheader("Referensi")
        st.write("Alahmar, A., Mohammed, E., & Benlamri, R. (2018). Application of Data Mining Techniques to Predict the Length of Stay of Hospitalized Patients with Diabetes. 2018 4th International Conference on Big Data Innovations and Applications (Innovate-Data), 38-43. https://doi.org/10.1109/Innovate-Data.2018.00013")
        st.write("Riya, N. J., Chakraborty, M., & Khan, R. (2024). Artificial Intelligence-Based Early Detection of Dengue Using CBC Data. IEEE Access, 12, 112355–112367. https://doi.org/10.1109/ACCESS.2024.344329")
        st.write("Shahid Ansari, Md., Jain, D., Harikumar, H., Rana, S., Gupta, S., Budhiraja, S., & Venkatesh, S. (2021). Identification of predictors and model for predicting prolonged length of stay in dengue patients. Health Care Management Science, 24(4), 786-798. https://doi.org/10.1007/s10729-021-09571-3")
        st.write("Wiratmadja, I. I., Salamah, S. Y., & Govindaraju, R. (2018). Healthcare Data Mining: Predicting Hospital Length of Stay of Dengue Patients. Journal of Engineering and Technological Sciences, 50(1), 110-126. https://doi.org/10.5614/j.eng.technol.sci.2018.50.1.8")
        st.write("Wulandari, D. A. P., Permana, K. A. B., & Sudarma, M. (2018). Prediction of Days in Hospital Dengue Fever Patients using K-Nearest Neighbor. International Journal of Engineering and Emerging Technology, 3(1), 23-25.")

    with tab2:

        st.subheader("2.1 Load Dataset")
        st.write("Berikut merupakan code yang digunakan untuk melakukan load dataset pada python")
        code = '''df = pd.read_csv("./assets/resource/dengue_fever_los_dataset.csv")'''
        st.code(code, language="python")
        st.write("Output: ")

        data = pd.read_csv("./assets/resource/dengue_fever_los_dataset.csv")
        st.write(data) 

        st.subheader("2.2 Data Transformation")
        st.write("Berikut merupakan code yang digunakan untuk melakukan data transformation pada python")

        data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
        data['jenis_demam'] = data['jenis_demam'].map({'DSS': 2, 'DBD': 1, 'DD': 0})
        median_value = data['lama_dirawat'].median()
        data['kategori_lama_dirawat'] = data['lama_dirawat'].apply(lambda x: 1 if x > median_value else 0)
        
        code = '''
        data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
        data['jenis_demam'] = data['jenis_demam'].map({'DSS': 2, 'DBD': 1, 'DD': 0})
        median_value = data['lama_dirawat'].median()
        data['kategori_lama_dirawat'] = data['lama_dirawat'].apply(lambda x: 1 if x > median_value else 0)
        '''
        st.code(code, language="python")
        st.write("Output: ", data)

        st.subheader("2.3 Imputasi Missing Value")
        st.write("Berikut merupakan code yang digunakan untuk melakukan data transformation pada python")

        # missing_values = data.isnull().sum()

        code = '''
        # membuat fungsi untuk imputasi missing value
        def impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5):
            imputer = KNNImputer(n_neighbors=n_neighbors)
            cols_reference_numeric = data[cols_reference].select_dtypes(include=['float64', 'int64']).columns.tolist()
            imputed_values = imputer.fit_transform(data[cols_reference_numeric + cols_to_impute])
            data[cols_to_impute] = imputed_values[:, len(cols_reference_numeric):]
            return data
        
        # pilih kolom yang akan dilakukan imputasi
        cols_to_impute = ['hct', 'hemoglobin']
        cols_reference = ['jenis_kelamin', 'jenis_demam', 'trombosit', 'kategori_lama_dirawat', 'hct', 'hemoglobin']

        #lakukan imputasi menggunakan function yang telah dibuat
        impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5)
        data.head()
        '''
        st.code(code, language="python")
        # st.text(missing_values)

        def impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5):
            imputer = KNNImputer(n_neighbors=n_neighbors)
            cols_reference_numeric = data[cols_reference].select_dtypes(include=['float64', 'int64']).columns.tolist()
            imputed_values = imputer.fit_transform(data[cols_reference_numeric + cols_to_impute])
            data[cols_to_impute] = imputed_values[:, len(cols_reference_numeric):]
            return data
        
        # pilih kolom yang akan dilakukan imputasi
        cols_to_impute = ['hct', 'hemoglobin']
        cols_reference = ['jenis_kelamin', 'jenis_demam', 'trombosit', 'kategori_lama_dirawat', 'hct', 'hemoglobin']

        #lakukan imputasi menggunakan function yang telah dibuat
        impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5)

        st.write("Output: ", data)

        st.subheader("2.3 Minmax Normalization")
        st.write("Berikut merupakan code yang digunakan untuk melakukan normalisasi data pada python")

        # missing_values = data.isnull().sum()

        code = '''
        # membuat fungsi untuk imputasi missing value
        def normalize_columns(data, cols_to_normalize):
            scaler = MinMaxScaler()
            data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
            return data
        
        #pilih kolom yang akan dinormalisasi menggunakan fungsi sebelumnya
        normalize_columns(data, ['umur', 'hct', 'hemoglobin', 'trombosit'])
        '''
        st.code(code, language="python")

        def normalize_columns(data, cols_to_normalize):
            scaler = MinMaxScaler()
            data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
            # joblib.dump(scaler, /content/drive/MyDrive/Task/Semester 8/Dataset/)
            return data
        
        normalize_columns(data, ['umur', 'hct', 'hemoglobin', 'trombosit'])

        st.write("Output: ", data)

        st.subheader("2.4 Menghapus Fitur")
        st.write("Berikut merupakan code yang digunakan untuk menghapus fitur yang tidak diperlukan pada python")

        code = '''
        # membuat fungsi untuk remove columns
        def remove_columns(data, columns_to_remove):
            data = data.drop(columns=columns_to_remove)
            return data

        # menghapus fitur menggunakan fungsi
        data_final = remove_columns(data, ['tgl_masuk', 'tgl_keluar', 'rm', 'lama_dirawat'])
        '''
        st.code(code, language="python")

        def remove_columns(data, columns_to_remove):
            data = data.drop(columns=columns_to_remove)
            return data
        
        data_final = remove_columns(data, ['tgl_masuk', 'tgl_keluar', 'rm', 'lama_dirawat'])
        data_final.head()

        st.write("Output: ", data_final)

        st.subheader("2.5 Membagi Dataset")
        st.write("Berikut merupakan code yang digunakan untuk membagi dataset menjadi 70% training dan 30% testing pada python")

        code = '''
        # membuat fungsi untuk remove columns
        train_data, test_data = train_test_split(data_final, test_size=0.3, random_state=42)

        # melihat hasil pembagian data
        print(train_data.shape)
        print(test_data.shape)

        '''

        train_data, test_data = train_test_split(data_final, test_size=0.3, random_state=42)

        st.code(code, language="python")

        st.write("Output: ")
        st.write("Hasil data training: ", train_data.shape)
        st.write("Hasil data test: ", test_data.shape)

        st.subheader("2.7 Metode Tunggal")
        st.subheader("2.7.1 K-Nearest Neighbord (KNN)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan KNN pada python")

        code = '''
        # menggunakan model knn pada sklearn
        knn = KNeighborsClassifier()

        # menentukan parameter yang akan di tuning
        parameters_knn = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'metric': ['euclidean']
        }

        # melakukan grid search pada knn
        grid_search_knn = GridSearchCV(knn, parameters_knn, cv=kf, scoring='accuracy')
        grid_search_knn.fit(X_train, y_train)

        # menampilkan hasil parameter terbaik
        print(grid_search_knn.best_score_)
        print(grid_search_knn.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_knn = grid_search_knn.best_estimator_
        y_pred_knn = best_model_knn.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil terbaik: 0.6509614015097565 Parameter: {'metric': 'euclidean', 'n_neighbors': 9}")

        st.subheader("2.7.2 Logistic Regression (LR)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan LR pada python")

        code = '''
        # menggunakan model lr pada sklearn
        lr = LogisticRegression()

        # menentukan parameter yang akan di tuning
        parameters_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        # melakukan grid search pada lr
        grid_search_lr = GridSearchCV(lr, parameters_lr, cv=kf, scoring='accuracy')
        grid_search_lr.fit(X_train, y_train)

        # menampilkan hasil parameter terbaik
        print(grid_search_lr.best_score_)
        print(grid_search_lr.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_lr = grid_search_lr.best_estimator_
        y_pred_lr = best_model_lr.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil terbaik: 0.6373735935051987 Paremeter: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}")

        st.subheader("2.7.3 Support Vector Machine (SVM)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan SVM pada python")

        code = '''
        # menggunakan model svm pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_svm = GridSearchCV(svm, parameters_svm, cv=kf, scoring='accuracy')
        grid_search_svm.fit(X_train, y_train)

        # menampilkan hasil parameter terbaik
        print(grid_search_svm.best_score_)
        print(grid_search_svm.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_svm = grid_search_svm.best_estimator_
        y_pred_svm = best_model_svm.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6458481697763852 Parameter: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}")

        st.subheader("2.7.3 Artificial Neural Network (ANN)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan ANN pada python")

        code = '''
        # menggunakan model ann pada sklearn
        mlp = MLPClassifier()

        # menentukan parameter yang akan di tuning
        parameters_mlp = {
            'max_iter': [100, 500, 1000],
            'learning_rate_init'   : [0.0001, 0.001, 0.01, 0.1],
            'activation': ['tanh', 'relu', 'logistic'],
            'hidden_layer_sizes' : [(i,) for i in range(1, 11)]
        }

        # melakukan grid search pada ann
        grid_search_mlp = GridSearchCV(mlp, parameters_mlp, cv=kf, scoring='accuracy')
        grid_search_mlp.fit(X_train, y_train)
        

        # menampilkan hasil parameter terbaik
        print(grid_search_mlp.best_score_)
        print(grid_search_mlp.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_mlp = grid_search_mlp.best_estimator_
        y_pred_mlp = best_model_mlp.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.644195983478137 Parameter: {'activation': 'relu', 'hidden_layer_sizes': (4,), 'learning_rate_init': 0.1, 'max_iter': 1000}")

        st.subheader("2.8 Metode Ensemble Stacking")
        st.write("Untuk membuat model ensemble stacking langkah pertama adalah menyimpan masing-masing hasil prediksi pada model.")

        # table_data_desc = pd.DataFrame(
        #     [
        #         {"Data": "Prediksi validasi metode tunggal", "Keterangan": "Prediksi validasi metode tunggal pada masing-masing fold saat traning disimpan untuk digunakan pelatihan pada model Stacking"},
        #         {"Data": "Prediksi testing metode tunggal", "Keterangan": "Prediksi testing metode tunggal pada saat testing disimpan untuk digunakan testing pada model Stacking"},
        #     ]
        # )

        st.write("Keterangan")
        st.write("Prediksi validasi metode tunggal: Prediksi validasi metode tunggal pada masing-masing fold saat traning disimpan untuk digunakan pelatihan pada model Stacking")
        st.write("Prediksi testing metode tunggal: Prediksi testing metode tunggal pada saat testing disimpan untuk digunakan testing pada model Stacking")

        st.write("Berikut merupakan code yang digunakan untuk mengumpulkan hasil prediksi pada data validasi metode tunggal")

        code = '''
        # menyimpan beberapa model yang digunakan pada metode tunggal
        models = [
            ('knn_model_optimize', KNeighborsClassifier, grid_search_knn),
            ('lr_model_optimize', LogisticRegression, grid_search_lr),
            ('svm_model_optimize', SVC, grid_search_svm),
            ('mlp_model_optimize', MLPClassifier, grid_search_mlp)
        ]

        # menyiapkan list untuk hasil prediksi dan label aktual
        predictions = {model[0]: [] for model in models}
        y_val_actuals = []
        test_val_indexs = []

        for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train), 1):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            for model_name, model_class, grid_search in models:
                model = model_class(**grid_search.best_params_)
                model.fit(X_train_fold, y_train_fold)
                y_pred_val = model.predict(X_val_fold)
                predictions[model_name].extend(y_pred_val)

            y_val_actuals.extend(y_val_fold.values)
            test_val_indexs.extend(test_index)
        
        #hasil prediksi disimpan pada dataframe
        results_validation = pd.DataFrame({
            'test_index': test_val_indexs,
            'y_pred_knn': predictions['knn_model_optimize'],
            'y_pred_lr': predictions['lr_model_optimize'],
            'y_pred_svm': predictions['svm_model_optimize'],
            'y_pred_mlp': predictions['mlp_model_optimize'],
            'y_actual': y_val_actuals
        })

        '''

        st.code(code, language="python")

        st.write("Berikut merupakan code yang digunakan untuk mengumpulkan hasil prediksi pada data testing metode tunggal")

        code = '''
        # data testing sebelumnya sudah disimpan pada variabel y_pred, kemudian sekarang dikumpulkan menjadi 1
        results_testing = pd.DataFrame({
            'y_pred_knn'  : y_pred_knn,
            'y_pred_lr'   : y_pred_lr,
            'y_pred_svm'  : y_pred_svm,
            'y_pred_mlp'  : y_pred_mlp,
            'y_actual'    : y_test
        })
        '''

        st.code(code, language="python")

        st.subheader("2.7.1 Meta-Classifier KNN")
        st.write("Untuk membuat KNN sebagai meta-classifier langkah awalnya adalah menghapus y_pred_knn dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_knn dari dataset training
        ensemble_train_knn = results_validation.drop('y_pred_knn', axis=1)

        # menghapus y_pred_knn dari dataset testing
        ensemble_test_knn = results_testing.drop('y_pred_knn', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_knn, y_ensemble_train_knn = ensemble_train_knn.drop(['y_actual', 'test_index'], axis=1), ensemble_train_knn['y_actual']
        X_ensemble_test_knn, y_ensemble_test_knn = ensemble_test_knn.drop('y_actual', axis=1), ensemble_test_knn['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        knn = KNeighborsClassifier()

        # menentukan parameter yang akan di tuning
        parameters_knn = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'metric': ['euclidean']
        }

        # melakukan grid search pada knn
        grid_search_ensemble_knn = GridSearchCV(knn, parameters_knn, cv=kf, scoring='accuracy')
        grid_search_ensemble_knn.fit(X_ensemble_train_knn, y_ensemble_train_knn)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_knn.best_score_)
        print(grid_search_ensemble_knn.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_knn = grid_search_ensemble_knn.best_estimator_
        y_pred_ensemble_knn = best_model_ensemble_knn.predict(X_ensemble_test_knn)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6390827517447657 Parameter: {'metric': 'euclidean', 'n_neighbors': 3}")

        st.subheader("2.7.2 Meta-Classifier Logistic Regression (LR)")
        st.write("Untuk membuat LR sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_lr dari dataset training
        ensemble_train_lr = results_validation.drop('y_pred_lr', axis=1)

        # menghapus y_pred_lr dari dataset testing
        ensemble_test_lr = results_testing.drop('y_pred_lr', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_lr, y_ensemble_train_lr = ensemble_train_lr.drop(['y_actual', 'test_index'], axis=1), ensemble_train_lr['y_actual']
        X_ensemble_test_lr, y_ensemble_test_lr = ensemble_test_lr.drop('y_actual', axis=1), ensemble_test_lr['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        lr = LogisticRegression()

        # menentukan parameter yang akan di tuning
        parameters_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        # melakukan grid search pada lr
        grid_search_ensemble_lr = GridSearchCV(lr, parameters_lr, cv=kf, scoring='accuracy')
        grid_search_ensemble_lr.fit(X_ensemble_train_lr, y_ensemble_train_lr)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_lr.best_score_)
        print(grid_search_ensemble_lr.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_lr = grid_search_ensemble_lr.best_estimator_
        y_pred_ensemble_lr = best_model_ensemble_lr.predict(X_ensemble_test_lr)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6373735935051987 Parameter: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}")


        st.subheader("2.7.3 Meta-Classifier Support Vector Machine (SVM)")
        st.write("Untuk membuat SVM sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_svm dari dataset training
        ensemble_train_svm = results_validation.drop('y_pred_svm', axis=1)

        # menghapus y_pred_svm dari dataset testing
        ensemble_test_svm = results_testing.drop('y_pred_svm', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_svm, y_ensemble_train_svm = ensemble_train_svm.drop(['y_actual', 'test_index'], axis=1), ensemble_train_svm['y_actual']
        X_ensemble_test_svm, y_ensemble_test_svm = ensemble_test_svm.drop('y_actual', axis=1), ensemble_test_svm['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_ensemble_svm = GridSearchCV(svm, parameters_svm, cv=kf, scoring='accuracy')
        grid_search_ensemble_svm.fit(X_ensemble_train_svm, y_ensemble_train_svm)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_svm.best_score_)
        print(grid_search_ensemble_svm.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_svm = grid_search_ensemble_svm.best_estimator_
        y_pred_ensemble_svm = best_model_ensemble_svm.predict(X_ensemble_test_svm)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6508617006124484 Parameter: {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}")

        st.subheader("2.7.4 Meta-Classifier Artificial Neural Network (ANN)")
        st.write("Untuk membuat ANN sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_ann dari dataset training
        ensemble_train_ann = results_validation.drop('y_pred_ann', axis=1)

        # menghapus y_pred_ann dari dataset testing
        ensemble_test_ann = results_testing.drop('y_pred_ann', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_mlp, y_ensemble_train_mlp = ensemble_train_mlp.drop(['y_actual', 'test_index'], axis=1), ensemble_train_mlp['y_actual']
        X_ensemble_test_mlp, y_ensemble_test_mlp = ensemble_test_mlp.drop('y_actual', axis=1), ensemble_test_mlp['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_ensemble_mlp = GridSearchCV(mlp, parameters_mlp, cv=kf, scoring='accuracy')
        grid_search_ensemble_mlp.fit(X_ensemble_train_mlp, y_ensemble_train_mlp)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_mlp.best_score_)
        print(grid_search_ensemble_mlp.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_mlp = grid_search_ensemble_mlp.best_estimator_
        y_pred_ensemble_mlp = best_model_ensemble_mlp.predict(X_ensemble_test_mlp)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6542942600769122 Parameter: {'activation': 'tanh', 'hidden_layer_sizes': (4,), 'learning_rate_init': 0.001, 'max_iter': 500}")

        st.subheader("2.9 Evaluasi Model")
        st.write("Berikut merupakan code yang digunakan untuk melakukan evaluasi model")

        code = '''
        #membuat function untuk evaluasi model
        def evaluate_model(y_true, y_pred):
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precission": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1-score": f1_score(y_true, y_pred)
            }

        return metrics

        #melakukan evaluasi model menggunakan function
        eval_knn  = evaluate_model(y_test, y_pred_knn)
        eval_lr   = evaluate_model(y_test, y_pred_lr)
        eval_svm  = evaluate_model(y_test, y_pred_svm)
        eval_mlp  = evaluate_model(y_test, y_pred_mlp)

        #menyimpan hasil evaluasi
        metrics_base_model = {
            "KNN" : eval_knn,
            "LR"  : eval_lr,
            "SVM" : eval_svm,
            "ANN" : eval_mlp,
        }

        '''

        table_results = pd.DataFrame(
            [
                {"Model": "KNN", "Accuracy": 0.62, "Precission": 0.44, "Recall": 0.32, "F1-Score": 0.37},
                {"Model": "LR", "Accuracy": 0.65, "Precission": 0.52, "Recall": 0.17, "F1-Score": 0.25},
                {"Model": "SVM", "Accuracy": 0.67, "Precission": 0.69, "Recall": 0.12, "F1-Score": 0.21},
                {"Model": "ANN", "Accuracy": 0.65, "Precission": 0.52, "Recall": 0.18, "F1-Score": 0.26},
            ]
        )

        st.code(code, language="python")

        st.write("Output: ")

        st.dataframe(table_results, use_container_width=True)

        st.write("Berikut merupakan code yang digunakan untuk melakukan evaluasi model ensemble")

        code = '''
        
        #melakukan evaluasi model menggunakan function
        eval_ensemble_knn = evaluate_model(y_test, y_pred_ensemble_knn)
        eval_ensemble_lr  = evaluate_model(y_test, y_pred_ensemble_lr)
        eval_ensemble_svm = evaluate_model(y_test, y_pred_ensemble_svm)
        eval_ensemble_mlp = evaluate_model(y_test, y_pred_ensemble_mlp)

        #menyimpan hasil evaluasi
        metrics_ensemble = {
            "Ensemble KNN" : eval_ensemble_knn,
            "Ensemble LR" : eval_ensemble_lr,
            "Ensemble SVM" : eval_ensemble_svm,
            "Ensemble ANN" : eval_ensemble_mlp
        }

        '''

        table_results_ensemble = pd.DataFrame(
            [
                {"Model": "KNN", "Accuracy": 0.66, "Precission": 0.65, "Recall": 0.27, "F1-Score": 0.36},
                {"Model": "LR", "Accuracy": 0.66, "Precission": 0.56, "Recall": 0.17, "F1-Score": 0.26},
                {"Model": "SVM", "Accuracy": 0.62, "Precission": 0.44, "Recall": 0.32, "F1-Score": 0.37},
                {"Model": "ANN", "Accuracy": 0.65, "Precission": 0.53, "Recall": 0.10, "F1-Score": 0.17},
            ]
        )

        st.code(code, language="python")

        st.write("Output: ")

        st.dataframe(table_results_ensemble, use_container_width=True)

        






if __name__ == '__main__':
    main()