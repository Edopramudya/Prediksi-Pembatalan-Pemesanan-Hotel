import pandas as pd
import os

# Menentukan path file dataset yang ada di folder yang sama
input_path = r"C:\Users\User\Documents\semester 3\kecerdasan artifisial\uas\dataset 1000.csv"
output_path = r"C:\Users\User\Documents\semester 3\kecerdasan artifisial\uas\dataset_ai2024.csv"

# Membaca dataset
df = pd.read_csv(input_path)

# Menampilkan beberapa baris pertama sebelum penghapusan kolom
print("Data sebelum penghapusan kolom:")
print(df.head())

# Menghapus kolom "Booking_ID" dan "date of reservation"
columns_to_drop = ["Booking_ID", "date of reservation"]
df.drop(columns=columns_to_drop, inplace=True)

# Menampilkan beberapa baris pertama setelah penghapusan kolom
print("\nData setelah penghapusan kolom:")
print(df.head())

# Menyimpan hasil ke file baru
df.to_csv(output_path, index=False)

print(f"\nDataset berhasil disimpan sebagai: {output_path}")
