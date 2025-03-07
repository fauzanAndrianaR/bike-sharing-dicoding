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


# Visualisasi Data
st.subheader("Tren Penyewaan Sepeda Harian")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x=day_df['dteday'], y=day_df['cnt'], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)



st.write("")
st.write("")
st.write("## Pertanyaan 1 : Bagaimana pengaruh faktor cuaca terhadap jumlah penyewaan sepeda dalam setiap harinya?")

st.subheader("Pengaruh Cuaca terhadap Penyewaan Sepeda")


# Menghitung total penyewaan sepeda berdasarkan musim
season_rentals = day_df.groupby('season')['cnt'].sum().reset_index()

# Menyesuaikan label musim
season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
season_rentals['season'] = season_rentals['season'].map(season_labels)

# Streamlit UI
st.write("Total Penyewaan Sepeda Berdasarkan Musim")

# Menampilkan tabel di Streamlit
st.dataframe(season_rentals)

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
st.markdown("> Penyewaan sepeda tertinggi terjadi pada musim gugur, kemungkinan karena cuaca yang lebih nyaman untuk bersepeda, sementara penyewaan terendah terjadi pada musim semi, mungkin akibat cuaca yang masih tidak stabil. Musim panas dan musim dingin memiliki jumlah penyewaan yang cukup tinggi, tetapi tidak setinggi musim gugur. Secara keseluruhan, tren ini menunjukkan bahwa faktor cuaca berpengaruh terhadap jumlah penyewaan sepeda, dengan kondisi yang lebih hangat dan stabil cenderung meningkatkan minat masyarakat untuk bersepeda.")
st.markdown("> Berdasarkan hasil Clustering, Kondisi cuaca yang ekstrem, seperti suhu yang sangat dingin atau panas berlebih, serta kelembapan tinggi, menyebabkan penurunan jumlah penyewaan.")
st.markdown("> Faktor cuaca berpengaruh signifikan terhadap jumlah penyewaan sepeda. Penyewaan tertinggi terjadi saat cuaca cerah karena kondisi yang nyaman. Saat cuaca mendung atau berkabut, penyewaan sedikit menurun akibat visibilitas rendah dan udara lembab. Pada cuaca buruk seperti hujan atau salju ringan, penyewaan berkurang drastis karena jalan licin. Dalam kondisi ekstrem seperti hujan deras atau salju tebal, penyewaan hampir tidak ada, menunjukkan bahwa pengguna lebih memilih transportasi lain saat cuaca sangat buruk. ")
st.markdown("**Conclusion:**")
st.markdown("> Cuaca yang baik mendorong lebih banyak orang untuk menyewa sepeda, sementara cuaca buruk atau ekstrem secara drastis mengurangi jumlah penyewaan. Hal ini bisa menjadi pertimbangan bagi penyedia layanan dalam merencanakan jumlah sepeda yang tersedia dan strategi promosi pada berbagai kondisi cuaca.")

st.write("# ")
st.write("")
st.write("")
st.write("")


# Menghitung rata-rata penyewaan per jam
st.write("## Pertanyaan 2 : Pada jam berapa jumlah penyewaan sepeda tertinggi dan terendah dalam sehari dan apa saja faktor yang mempengaruhinya?")
hourly_rentals = hour_df.groupby('hr')['cnt'].mean()

st.subheader("Tren Penyewaan Sepeda dalam Sehari")
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
st.markdown("> Pola penyewaan sepeda menunjukkan lonjakan pada pagi (07:00-09:00) dan sore (17:00-19:00), mencerminkan penggunaan utama sebagai transportasi kerja atau sekolah. Penyewaan terendah terjadi pada dini hari (00:00-05:00) karena aktivitas berkurang. Siang hari (10:00-16:00) memiliki tingkat penyewaan stabil, didominasi oleh pengguna rekreasi atau aktivitas santai.")
st.markdown("**Conclusion:**")
st.markdown("> Penyewaan sepeda memiliki pola dua puncak utama, yaitu saat pagi dan sore hari, yang berkaitan dengan aktivitas kerja dan sekolah. Sementara itu, pada dini hari, jumlah penyewaan sangat rendah. Hal ini dapat membantu penyedia layanan dalam mengatur jumlah sepeda yang tersedia pada jam-jam sibuk dan mengoptimalkan strategi operasional mereka.")


st.write("# ")
st.write("")
st.write("")
st.write("")

# Kesimpulan
st.subheader("Kesimpulan")
st.write("Dari analisis ini, dapat disimpulkan bahwa tren penyewaan sepeda dipengaruhi oleh berbagai faktor seperti cuaca, suhu, musim dan waktu. Dengan pemahaman ini, pengelola layanan penyewaan sepeda dapat mengoptimalkan strategi operasional mereka untuk meningkatkan layanan dan pendapatan.")