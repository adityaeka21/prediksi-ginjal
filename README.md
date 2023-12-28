# Laporan Proyek Machine Learning

### Nama : Aditya Eka Purwanto

### Nim : 211351005

### Kelas : Malam B

## Domain Proyek

Web app ini ditujukan untuk membantu tenaga medis, seperti dokter atau perawat, dalam melakukan diagnosis yang lebih tepat dan cepat. Dengan menggunakan model prediksi, mereka dapat menentukan kemungkinan gagal ginjal pada pasien berdasarkan data yang dimasukkan, penting untuk diingat bahwa hasil prediksi diberikan hanya bersifat prediktif. Keputusan akhir mengenai diagnosis dan penanganan pasien harus didasarkan pada pertimbangan menyeluruh yang mencakup data medis lengkap.

## Business Understanding

Tujuan dari Web app ini adalah untuk meningkatkan akurasi diagnosis dan menghemat waktu tenaga medis dengan memberikan prediksi yang lebih baik. Dengan demikian, diharapkan dapat mengurangi kesalahan dalam diagnosis dan memungkinkan penanganan yang lebih cepat terhadap kondisi gagal ginjal.

### Problem Statements

Meskipun ada peningkatan dalam teknologi medis, diagnosis yang tepat dan efisien untuk kasus gagal ginjal masih menjadi tantangan. Kurangnya alat yang dapat memberikan prediksi akurat tentang risiko gagal ginjal berpotensi menyebabkan penundaan dalam penanganan dan perawatan yang tepat.

### Solution statements

Memberikan alat bagi tenaga medis untuk melakukan diagnosis lebih akurat dan cepat berdasarkan data yang tersedia, memungkinkan identifikasi dini serta pengelolaan lebih efisien terhadap kasus-kasus gagal ginjal

### Goals

Menghasilkan sebuah Web app yang dapat memberikan prediksi risiko gagal ginjal dengan tingkat akurasi yang tinggi. Dengan memanfaatkan teknologi machine learning dan data medis yang relevan.

## Data Understanding

Dataset yang saya gunakan ini berasal dari UCI Machine Learning Repository. Tujuan dari dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien mengalami penyakit ginjal kronis atau tidak, berdasarkan pada beberapa pengukuran diagnostik yang terdapat dalam dataset.
Dataset ini terdiri dari beberapa variabel prediktor medis dan satu variabel target, yakni Kelas. Variabel prediktor termasuk Tekanan Darah (Bp), Albumin (Al), dan lain-lain.
Datasets [chronic kidney disease](https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease).

### Variabel-variabel pada chronic kidney disease adalah sebagai berikut:

-   Bp :  
    Tekanan darah/Blood Pressure.  
    range 50-180 -- Tipe float64

-   Sg :
    Specific Gravity.  
    range 1.005-1.025 -- Tipe float64
-   Al
    Albumin g/dL.  
    range 0-5 -- Tipe float64
-   Su :
    Level kadar gula pada urine.  
    range 0-5 -- Tipe float64
-   Rbc :  
    Jumlah sel darah merah dalam urin.  
    range 0-1 -- Tipe float64
-   Bu :
    Blood urea dalam darah.  
    range 1.5-391.0 -- Tipe float64
-   Sc :
    Kadar serum kreatin dalam urine mg/dL.  
    range 0.4-76.0 -- Tipe float64
-   Sod :
    Kadar natrium dalam urine mEq/L.  
    range 4.5-163.0 -- Tipe float64
-   Pot :
    Kadar pottasium dalam urine.  
    range 2.5-47 -- Tipe float64
-   Hemo :
    Kadar Hemoglobin g/dL.  
    range 3.1-17.8 -- Tipe float64
-   Wbcc :
    Jumlah sel darah putih dalam darah.  
    range 2200-26400 -- Tipe float64
-   Rbcc :
    Jumlah sel darah merah dalam darah.  
    range 2.1-8.0 -- Tipe float64
-   Htn :
    Status pasien apakah mengidap hiper tensi atau tidak.  
    range 0-1 -- Tipe float64
-   Class :
    Status pasien apakah mengidap gagal ginjal kronis atau tidak.  
    range 0-1 -- Tipe int64

## Data Preparation

Pada proses ini, fokus saya adalah mengolah dan menganalisis dataset yang telah dipilih. Dataset ini telah melalui proses pembersihan dan seluruh kolomnya memiliki tipe data numerik yang sesuai untuk proses analisis yang akan dilakukan.

Pertama kita import file kaggle agar bisa mengunduh datasetnya

```py
from google.colab import files
files.upload()
```

Langkah selanjutnya membuat folder untuk menyimpan filenya

```py
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Lalu Mengunduh datasets

```py
!kaggle datasets download -d abhia1999/chronic-kidney-disease
```

Langkah berikutnya adalah membuat folder baru dan mengekstrak file yang telah diunduh ke dalam folder tersebut.

```py
!mkdir chronic-kidney-disease
!unzip chronic-kidney-disease.zip -d chronic-kidney-disease
!ls chronic-kidney-disease
```

### Import library

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics
import pickle
import warnings
warnings.simplefilter("ignore")
```

Membuat data frame dari data csv yang di download tadi

```py
df = pd.read_csv("chronic-kidney-disease/new_model.csv")
```

Melihat 5 data pertama pada dataset

```py
df.head()
```

Output dibawah kita bisa melihat jumlah data, nilai min, max dan sebagainya dari tiap-tiap kolom

```py
df.describe()
```

### Visualisasi

Petama mari kita lihat korelasi antar kolom dengan menggunakan heatmap, dapat kita lihat htn memilki korelasi paling tinggi dengan label diantara kolom yang lain

```py
plt.figure(figsize = (15, 8))

sns.heatmap(df.corr(), annot = True, linewidths = 2, linecolor = 'lightgrey')
plt.show()
```

![heatmap](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/37ed7ae0-0dca-4ef6-a94f-b304f9dba624)

Selanjutnya kita lihat distribusi data dari tiap-tiap feature

```py
features = df.columns[:-1]

num_rows = len(features) // 3 + (len(features) % 3 > 0)
num_cols = 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

if num_rows == 1:
    axes = [axes]

for i, feature in enumerate(features):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row][col]

    sns.distplot(df[feature], color='green', ax=ax)
    ax.set_title(f'Distribution of {feature}')

if len(features) < num_rows * num_cols:
    for i in range(len(features), num_rows * num_cols):
        fig.delaxes(axes[i // num_cols][i % num_cols])

fig.suptitle("Distribution of Data", y=1.02)
plt.tight_layout()
plt.show()
```

![distrib_feature](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/dd9c645c-e330-4574-a9ce-815654233a6b)


Disini saya membuat fungsi untuk membuat plot

```py
def violin(col):
    fig = px.violin(df, y=col, x="Class", color="Class", box=True, template = 'plotly_dark')
    return fig.show()

def kde(col):
    grid = sns.FacetGrid(df, hue="Class", height = 6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
```

Dapat kita lihat memiliki hipertensi memiliki kemungkinan lebih besar untuk tidak terkena ginjal kronis

![vio_htn](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/a5561248-46c6-4747-b051-f6f188c52ce9)


Kadar BU (Blood Urea) tinggi
memiliki risiko lebih tinggi untuk terkena ginjal kronis

![vio_bu](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/a49b2bc7-f975-4271-ac2b-72902362c561)

Berdasarkan grafik dibawah kadar hemogoblin rendah memiliki risiko lebih tinggi untuk mengalami ginjal kronis

![vio_hemo](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/0d10dbd3-1219-4de2-b9dd-dcf4b172f8aa)

Berdasarkan plot dibawah dapat disimpulkan dengan kadar Rbcc rendah memiliki risiko lebih tinggi untuk mengalami ginjal kronis

![kde_rbcc](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/2ab267ed-7a0b-417c-898e-800c20ab5a98)

Berbeda dengan Rbcc, Wbcc tinggi memiliki risiko lebih tinggi untuk mengalami ginjal kronis walaupun tidak terlalu signifikan

![kde_wbcc](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/f0653e91-57de-4d1d-ae03-997f49255051)

## Modeling

Pada tahap ini saya akan membuat feature dari semua kolom kecuali kolom Class yang dimana akan dijadikan sebagai label dengan perintah berikut

```
X = data.drop('Class', axis=1)
y = data['Class']
```

Dilanjut dengan melakukan train_test_split dengan test 20% dan sisanya akan dijadikan data train

```py
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

Lanjut membuat model dengan model decision tree dengan kriteria entropy

```py
Dtree = DecisionTreeClassifier(criterion='entropy',random_state=20)
Dtree.fit(X_train,y_train)
y_pred = Dtree.predict(X_test)
```

Print akurasi yang kita dapatkan dari model ini, hasilnya sangat memuaskan yaitu 100.0%

```py
print("Akurasi model decision tree = ", accuracy_score(y_test,y_pred) * 100)
```

## Visuliasasi hasil algortima

Kita lihat visualisasi dari model decission tree kita dengan perintah berikut

```py
feature = [col for col in df.columns if col != 'Class']
label = 'Class'
fig = plt.figure(figsize=(15,20))
_ = tree.plot_tree(Dtree,
                   feature_names=feature,
                   class_names=label,
                   filled=True)
```

![tree](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/e216c7e0-79da-47a7-8462-6c5693e098e9)


Model ini memulai dengan memeriksa apakah kadar hemoglobin (Hemo) pasien kurang dari atau sama dengan 12,95. Jika ya, maka model akan memeriksa apakah kadar serum kreatinin (Sc) pasien kurang dari atau sama dengan 1,15. Jika ya, maka model akan memprediksi bahwa pasien menderita gagal ginjal. Jika tidak, maka model akan memprediksi bahwa pasien tidak menderita gagal ginjal.

Kita coba prediksi berdasarkan decision tree, hasilnya benar Hemo sangat berpengaruh pada prediksi model ini

```py
Bp = 10
Al = 1
Su = 0
Rbc = 10
Bu = 5
Sod = 123
Pot = 4

Sc = 1.2  # Memiliki ginjal krnois
Hemo = 10 # Memiliki ginjal krnois

# Hemo = 13 # Sehat
# Sg = 1.5 # Sehat

Sg = 1

Wbcc = 6800
Rbcc = 5.4
Htn = 1

input_data = np.array([[Bp,Sg,Al,Su,Rbc,Bu,Sc,Sod,Pot,Hemo,Wbcc,Rbcc,Htn]])

prediction = Dtree.predict(input_data)
if(prediction[0] == 1):
  print("Memiliki ginjal kronis")
else:
  print("Sehat")
```

## Save model

Save model yang dimana akan kita gunakan untuk prediksi menggunakan streamlit

```
filename = 'kidney_dt.sav'
pickle.dump(Dtree, open(filename, 'wb'))
```

## Evaluation

Karena hasil yang diharapkan merupakan prediksi, kita akan menggunakan metrik evaluasi confusion matrix

Confusion matrix adalah alat evaluasi yang berguna ketika kita melakukan prediksi menggunakan model machine learning. Alasan utama untuk menggunakan confusion matrix adalah untuk mengevaluasi seberapa baik kinerja model kita terhadap kelas-kelas yang berbeda dalam data.

```py
y_pred = Dtree.predict(X_test)
# Menghitung confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# Visualisasi confusion matrix dengan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm", fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```

![matrix](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/907abd35-79d3-4076-836f-a1e7e4465fa5)

## Deployment

Web app : [Streamlit](https://prediksi-gagal-ginjal.streamlit.app/)

![deply](https://github.com/adityaeka21/prediksi-ginjal/assets/148531157/005dbf4b-992f-4a81-b48d-7771ae481f19)

