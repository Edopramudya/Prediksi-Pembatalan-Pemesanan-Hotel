import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)
# Load the pre-trained model
trees = joblib.load('random_forest_model.joblib')

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Streamlit UI
st.sidebar.title(" **Prediksi Pembatalan Hotel**")

# Add a logo or image to the sidebar
st.sidebar.image("logoai.png", width=150)

# Add description to sidebar
st.sidebar.markdown("""
 
    Di sini, Anda dapat mengeksplorasi data, memprediksi kemungkinan pembatalan, 
    dan mengoptimalkan tingkat hunian hotel Anda.
""")

# Sidebar navigation options with selectbox
page = st.sidebar.selectbox(
    "Pilih Halaman", 
    ("Latar Belakang", "Tujuan Proyek", "Prediksi Pembatalan", "Evaluasi Model", "Prediksi dengan Dataset"),
    index=0  # Default to "Latar Belakang"
)
if page == "Latar Belakang":
    # Judul Halaman
    st.title("🏨 Prediksi Pembatalan Pemesanan Hotel")
    st.markdown("### **Mengoptimalkan Tingkat Hunian Hotel Anda**")
    st.markdown("---")

    st.write("""
        Industri perhotelan menghadapi tantangan besar dalam mengatasi pembatalan pemesanan, yang sering kali 
        berdampak pada pendapatan dan perencanaan operasional. Dengan menggunakan pendekatan berbasis data, 
        hotel dapat memprediksi kemungkinan pembatalan pemesanan sehingga keputusan strategis dapat diambil untuk 
        meminimalkan kerugian dan meningkatkan efisiensi.
    """)
    
    
    # Membuat dua tab: Tips dan Call to Action
    tab1, tab2 = st.tabs(["💡 Tips", "🚀 Call to Action"])

    # Tab 1: Tips untuk Mengurangi Risiko Pembatalan
    with tab1:
        st.markdown("### **Tips untuk Mengurangi Risiko Pembatalan**")
        st.success(""" 
            - **Optimalkan Waktu Pemesanan**: Pemesanan yang lebih dekat dengan tanggal menginap cenderung memiliki risiko pembatalan lebih rendah.
            - **Tawarkan Deposit Penuh**: Deposit penuh membantu meningkatkan komitmen pelanggan terhadap pemesanan mereka.
            - **Kelola Permintaan Khusus**: Hindari menerima terlalu banyak permintaan khusus, karena hal ini dapat meningkatkan risiko pembatalan.
        """)

    # Tab 2: Call to Action
    with tab2:
        st.write(""" 
            Dengan solusi ini, Anda tidak hanya dapat meningkatkan pendapatan, tetapi juga menciptakan pengalaman tamu yang lebih baik.
            **Mulai prediksi pembatalan sekarang!**
        """)

elif page == "Tujuan Proyek":
    
    st.title("🏨 Prediksi Pembatalan Pemesanan Hotel")
  
    st.subheader("Tujuan Proyek")
    tab1, tab2 = st.tabs(["💡 Tujuan", "🚀 detail tujuan"])

    # Tab 1: Tips untuk Mengurangi Risiko Pembatalan
    with tab1:
        st.write("""1.Meningkatkan pendapatan""")
        st.write("""2.Mengoptimalkan perencanaan operasional""")
        st.write("""3.Meningkatkan pengalaman pelanggan""")
        st.write("**4.Membantu pengambilan keputusan strategis**")
        

    # Tab 2: Call to Action
    with tab2:
        st.write(""" 
            - Meningkatkan pendapatan: Dengan memprediksi pembatalan lebih awal, hotel dapat mengoptimalkan tingkat hunian dan mengurangi risiko kerugian
            - Mengoptimalkan perencanaan operasional: Hotel dapat merencanakan kebutuhan staf, makanan, dan logistik lainnya lebih efisien
            - Meningkatkan pengalaman pelanggan: Dengan mengetahui pola pembatalan, hotel dapat memberikan layanan yang lebih baik dan membuat strategi retensi pelanggan
            - Membantu pengambilan keputusan strategis: Informasi tentang kemungkinan pembatalan dapat digunakan oleh manajemen hotel untuk menyusun strategi penetapan harga yang lebih fleksibel dan sesuai dengan pola pasar.
        """)
    # st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
        - Pemesanan dengan waktu yang lebih dekat cenderung memiliki risiko pembatalan lebih rendah
        - Pemesanan dengan deposit penuh memiliki tingkat pembatalan yang lebih rendah
        - Permintaan khusus yang terlalu banyak dapat meningkatkan risiko pembatalan
    """) 
elif page == "Prediksi Pembatalan":
    st.title("🏨 Prediksi Pembatalan Pemesanan Hotel")
    
    st.markdown("""
    ### 📝 Panduan Penggunaan
    Isi semua informasi pemesanan di bawah ini untuk mendapatkan prediksi kemungkinan pembatalan.
    Sistem akan menganalisis data menggunakan model machine learning yang telah dilatih.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Tamu")
        adults = st.number_input("Jumlah Dewasa", 1, 10, 1)
        children = st.number_input("Jumlah Anak", 0, 10, 0)
        weekend_nights = st.number_input("Malam Akhir Pekan", 0, 7, 0)
        week_nights = st.number_input("Malam Hari Kerja", 0, 30, 0)
    
    with col2:
        st.subheader("Detail Pemesanan")
        meal_type = st.selectbox("Paket Makanan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2'])
        room_type = st.selectbox("Tipe Kamar", ['Room Type 1', 'Room Type 2', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
        lead_time = st.number_input("Lead Time (Hari)", 0, 365, 50)
        price = st.number_input("Harga per Malam", 0.0, 1000.0, 100.0)

    col3, col4 = st.columns(2)
    with col3:
        parking = st.selectbox("Parkir", ['Tidak Ada', 'Ada'])
        market_segment = st.selectbox("Segmen Pasar", ['Offline', 'Online', 'Corporate', 'Complementary', 'Aviation'])
        special_requests = st.number_input("Permintaan Khusus", 0, 10, 0)

    input_data = {
        'number of adults': adults,
        'number of children': children,
        'number of weekend nights': weekend_nights,
        'number of week nights': week_nights,
        'type of meal': {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2}[meal_type],
        'car parking space': {'Tidak Ada': 0, 'Ada': 1}[parking],
        'room type': {'Room Type 1': 1, 'Room Type 2': 2, 'Room Type 4': 4, 'Room Type 5': 5, 'Room Type 6': 6, 'Room Type 7': 7}[room_type],
        'lead time': lead_time,
        'market segment type': {'Offline': 0, 'Online': 1, 'Corporate': 2, 'Complementary': 3, 'Aviation': 4}[market_segment],
        'average price': price,
        'special requests': special_requests
    }

    if st.button('🔮 Prediksi Pembatalan', help="Klik untuk melihat hasil prediksi"):
        
        df = pd.DataFrame([input_data])
        predictions = [predict(tree, df.iloc[0]) for tree in trees]
        prediction = max(set(predictions), key=predictions.count)
        
        if prediction == 1:
            st.warning("🚨 Kemungkinan Besar Akan Membatalkan")
        else:
            st.success("✅ Kemungkinan Besar Tidak Membatalkan")

        # result_html = """
        #     <div class="prediction {}">
        #         {}
        #     </div>
        # """.format(
        #     'warning' if prediction == 1 else 'success',
        #     'Kemungkinan Besar Akan Membatalkan' if prediction == 1 else 'Kemungkinan Besar Tidak Membatalkan'
        # )
        # st.markdown(result_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Pemesanan dengan waktu yang lebih dekat cenderung memiliki risiko pembatalan lebih rendah
        - Pemesanan dengan deposit penuh memiliki tingkat pembatalan yang lebih rendah
        - Permintaan khusus yang terlalu banyak dapat meningkatkan risiko pembatalan
        """) 
elif page == "Evaluasi Model":
    st.title("🏨 Prediksi Pembatalan Pemesanan Hotel")
    st.markdown("---")
    st.header("Evaluasi Model Random Forest")

    test = pd.read_csv('test_data.csv')
    # Definisi data untuk testing
    X_test = test.drop(['booking status'], axis=1).values
    y_test = test['booking status'].values
    
    # Get predictions for test data
    test_predictions = []
    for row in X_test:
        predictions = [predict(tree, row) for tree in trees]
        prediction = max(set(predictions), key=predictions.count)
        test_predictions.append(prediction)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions)
    recall = recall_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)

    # Display metrics with improved formatting
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Akurasi", value=f"{accuracy:.2%}")
        st.metric(label="Presisi", value=f"{precision:.2%}")
    with col2:
        st.metric(label="Recall", value=f"{recall:.2%}")
        st.metric(label="F1-Score", value=f"{f1:.2%}")
    
    # Add explanation of metrics
    with st.expander("Penjelasan Metrik Evaluasi"):
        st.write("""
        - **Akurasi**: Persentase prediksi yang benar dari total prediksi
        - **Presisi**: Persentase prediksi positif yang benar
        - **Recall**: Persentase kasus positif yang berhasil diprediksi
        - **F1-Score**: Rata-rata harmonik dari presisi dan recall
        """)
        st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
        - Pemesanan dengan waktu yang lebih dekat cenderung memiliki risiko pembatalan lebih rendah
        - Pemesanan dengan deposit penuh memiliki tingkat pembatalan yang lebih rendah
        - Permintaan khusus yang terlalu banyak dapat meningkatkan risiko pembatalan
    """) 
elif page == "Prediksi dengan Dataset":
    st.title("Prediksi Pembatalan Pemesanan dengan Dataset")
    
    # Membaca dataset lokal
    try:
        # Path dataset lokal
        dataset = pd.read_csv("encoded_data.csv")
        
        # Tampilkan dataset asli (opsional)
        st.write("Dataset Asli:")
        st.dataframe(dataset.head())

        X_test = dataset.drop(['booking status'], axis=1).values
        actual = dataset['booking status'].values
        
        test_predictions = []
        for row in X_test:
            predictions = [predict(tree, row) for tree in trees]
            prediction = max(set(predictions), key=predictions.count)
            test_predictions.append(prediction)

        # Tambahkan kolom prediksi dan aktual ke dataset
        dataset['Actual'] = actual
        dataset['Prediksi'] = test_predictions
        dataset['Prediksi Teks'] = dataset['Prediksi'].map({1: 'Pembatalan Terjadi', 0: 'Pembatalan Tidak Terjadi'})
        dataset['Actual Teks'] = dataset['Actual'].map({1: 'Pembatalan Terjadi', 0: 'Pembatalan Tidak Terjadi'})

        # Menampilkan hasil prediksi dan nilai aktual
        st.write("Hasil Prediksi vs Nilai Aktual:")
        st.dataframe(dataset[['Actual Teks', 'Prediksi Teks']].head(100))

        # Menyediakan opsi untuk mengunduh hasil
        st.write("Unduh hasil prediksi:")
        output_csv = dataset[['Actual Teks', 'Prediksi Teks']]
        csv = output_csv.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil sebagai CSV",
            data=csv,
            file_name='hasil_prediksi_vs_aktual.csv',
            mime='text/csv',
        )

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
            - Pemesanan dengan waktu yang lebih dekat cenderung memiliki risiko pembatalan lebih rendah
            - Pemesanan dengan deposit penuh memiliki tingkat pembatalan yang lebih rendah
            - Permintaan khusus yang terlalu banyak dapat meningkatkan risiko pembatalan
        """) 
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

