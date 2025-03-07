import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Header
st.set_page_config(page_title="Dashboard Bike Sharing", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: darkblue;'>Dashboard Analisis Data - Bike Sharing</h1>
    <hr>
""", unsafe_allow_html=True)

# Memuat dataset
day_df = pd.read_csv("day_cleaned.csv")
hour_df = pd.read_csv("hour_cleaned.csv")

# Konversi tanggal
day_df['dteday'] = pd.to_datetime(day_df['dteday'], errors='coerce')
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'], errors='coerce')

# Filter rentang waktu
st.sidebar.header("Filter Rentang Waktu")
start_date = st.sidebar.date_input("Tanggal Mulai", day_df['dteday'].min())
end_date = st.sidebar.date_input("Tanggal Akhir", day_df['dteday'].max())

if start_date > end_date:
    st.sidebar.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
else:
    day_df = day_df[(day_df['dteday'] >= pd.Timestamp(start_date)) & (day_df['dteday'] <= pd.Timestamp(end_date))]


st.write("")
st.write("")
st.write("## Pertanyaan 1 : - Bagaimana pengaruh faktor cuaca terhadap rata-rata jumlah penyewaan sepeda dalam setiap harinya? ")

st.subheader("Pengaruh Cuaca terhadap Penyewaan Sepeda")


st.write("")
st.write("")



st.write("### Clustering")

# Pilih fitur untuk clustering
features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
X = day_df[features]

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar untuk memilih jumlah cluster
k = st.sidebar.slider("Pilih jumlah cluster", 2, 10, 3)

# Terapkan K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
day_df['Cluster'] = kmeans.fit_predict(X_scaled)


# Plot clusters
st.write("### Visualisasi Cluster")
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(day_df['temp'], day_df['cnt'], c=day_df['Cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Temperature')
ax.set_ylabel('Total Rentals (cnt)')
ax.set_title('Clusters of Bike Rentals')
plt.colorbar(scatter, label='Cluster')
st.pyplot(fig)

# Tampilkan hasil clustering
st.write("### Distribusi Cluster")
st.write(day_df.groupby('Cluster')[features].mean())


# Menghitung rata-rata jumlah penyewaan berdasarkan kondisi cuaca
st.write("### Rata-rata jumlah penyewaan berdasarkan kondisi cuaca")

weather_rentals = day_df.groupby('weathersit', as_index=False)['cnt'].mean()

# Dictionary label kondisi cuaca
weathersit_labels = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}

# Pastikan semua kategori ada, meskipun tidak ada di dataset
all_weathersit = pd.DataFrame({'weathersit': [1, 2, 3, 4], 'cnt': 0})
weather_rentals = pd.concat([weather_rentals, all_weathersit]).drop_duplicates(subset=['weathersit'], keep='first')

# Ubah angka menjadi label kategori
weather_rentals['weathersit'] = weather_rentals['weathersit'].map(weathersit_labels)
weather_rentals = weather_rentals.set_index('weathersit').reindex(weathersit_labels.values()).reset_index()

# Gradasi warna biru
blue_shades = sns.color_palette("Blues", len(weather_rentals))[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=weather_rentals, x='weathersit', y='cnt', hue='weathersit', palette=blue_shades, legend=False, ax=ax)
ax.set_xlabel('Weather Condition')
ax.set_ylabel('Average Bike Rentals')
ax.set_title('Pengaruh Cuaca terhadap Penyewaan Sepeda')
st.pyplot(fig)

st.markdown("**Insight:**")
st.write("""
- **Cuaca Cerah (Clear, Few Clouds, Partly Cloudy) → Penyewaan Tertinggi**  
  Pada kondisi cuaca yang baik, jumlah penyewaan sepeda mencapai angka tertinggi. Ini menunjukkan bahwa pengguna cenderung lebih nyaman bersepeda saat cuaca mendukung.

- **Cuaca Mendung atau Berkabut (Mist, Cloudy) → Penyewaan Menurun**  
  Terjadi sedikit penurunan jumlah penyewaan dibandingkan dengan hari yang cerah. Kemungkinan disebabkan oleh visibilitas yang lebih rendah dan udara yang lebih lembab.

- **Cuaca Buruk (Hujan Ringan, Salju Ringan) → Penyewaan Berkurang Drastis**  
  Penyewaan turun signifikan karena kondisi jalan yang licin dan kurang nyaman bagi pesepeda.

- **Cuaca Ekstrem (Hujan Deras, Salju Tebal) → Penyewaan Paling Rendah**  
  Hampir tidak ada aktivitas penyewaan sepeda dalam kondisi ini.  
  Hal ini menunjukkan bahwa pengguna lebih memilih alternatif transportasi lain saat cuaca sangat buruk.
""")
st.markdown("> Berdasarkan hasil Clustering, Kondisi cuaca yang ekstrem, seperti suhu yang sangat dingin atau panas berlebih, serta kelembapan tinggi, menyebabkan penurunan jumlah penyewaan.")
st.markdown("**Conclusion:**")
st.markdown("> Cuaca yang baik mendorong lebih banyak orang untuk menyewa sepeda, sementara cuaca buruk atau ekstrem secara drastis mengurangi jumlah penyewaan. Hal ini bisa menjadi pertimbangan bagi penyedia layanan dalam merencanakan jumlah sepeda yang tersedia dan strategi promosi pada berbagai kondisi cuaca.")

st.write("# ")
st.write("")
st.write("")
st.write("")


# Menghitung rata-rata penyewaan per jam
st.write("## Pertanyaan 2 :- Pada jam berapa rata-rata jumlah penyewaan sepeda tertinggi dan terendah setiap harinya  dan apa saja faktor yang mempengaruhinya?")
hourly_rentals = hour_df.groupby('hr')['cnt'].mean()

st.subheader("Tren Rata-rata Penyewaan Sepeda dalam Sehari")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=hourly_rentals.index, y=hourly_rentals.values, marker='o', color='b', ax=ax)
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Average Bike Rentals')
ax.set_title('Tren Penyewaan Sepeda dalam Sehari')
ax.set_xticks(range(0, 24))
ax.grid(True)
st.pyplot(fig)

# Menentukan jam dengan penyewaan tertinggi dan terendah
peak_hour = hourly_rentals.idxmax()
low_hour = hourly_rentals.idxmin()
peak_value = hourly_rentals.max()
low_value = hourly_rentals.min()

# Score card penyewaan tertinggi & terendah
col1, col2 = st.columns(2)
with col1:
    st.metric(label="⏰ Jam Tertinggi", value=f"{peak_hour}:00", delta=int(peak_value))
with col2:
    st.metric(label="⏳ Jam Terendah", value=f"{low_hour}:00", delta=int(low_value))

st.markdown("**Insight:**")
st.write("""
- **Berdasarkan analisis data rata-rata penyewaan sepeda per jam, terdapat pola yang jelas dalam penggunaan sepeda sepanjang hari:**

- **Jam Penyewaan Tertinggi:** Pagi (07:00 - 09:00) dan Sore (17:00 - 19:00)  
  Lonjakan penyewaan sepeda diperkirakan terjadi saat jam berangkat kerja/sekolah dan jam pulang.  
  Hal ini menunjukkan bahwa banyak pengguna memanfaatkan sepeda sebagai alat transportasi utama pada hari kerja.

- **Jam Penyewaan Terendah:** Dini Hari (00:00 - 05:00)  
  Aktivitas penyewaan sepeda sangat rendah pada jam-jam ini.  
  Kemungkinan karena kebanyakan orang sudah beristirahat dan kondisi jalan yang lebih sepi.

- **Pola Siang Hari (10:00 - 16:00)**  
  Penyewaan tetap ada, tetapi lebih rendah dibandingkan jam sibuk.  
  Didominasi oleh pengguna yang bersepeda untuk rekreasi atau aktivitas santai.
""")

st.markdown("**Conclusion:**")
st.markdown("> Penyewaan sepeda memiliki pola dua puncak utama, yaitu saat pagi dan sore hari, yang berkaitan dengan aktivitas kerja dan sekolah. Sementara itu, pada dini hari, jumlah penyewaan sangat rendah. Hal ini dapat membantu penyedia layanan dalam mengatur jumlah sepeda yang tersedia pada jam-jam sibuk dan mengoptimalkan strategi operasional mereka.")

