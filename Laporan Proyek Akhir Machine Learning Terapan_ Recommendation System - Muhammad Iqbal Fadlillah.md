# Laporan Proyek Akhir Machine Learning Terapan: _Recommendation System_ - Muhammad Iqbal Fadlillah

## Project Overview
Pada proyek ini saya mengambil _project overview_ dalam lingkup topik rekomendasi buku. Judul dari proyek ini adalah sistem rekomendasi judul buku dengan metode _Content Based Filtering_ dan _Collaborative Filtering_. Tujuan dari proyek ini adalah memberikan rekomendasi buku kepada user berdasarkan atribut dari buku itu sendiri dan dari pengalaman user lain terhadap suatu buku. Latar belakang saya mengambil judul tersebut dikarenakan setiap orang mempunyai minat terhadap buku yang berbeda-beda sesuai dengan genre buku itu sendiri selain itu dengan adanya sistem rekomendasi buku dapat mempermudah para pembaca buku untuk menemukan buku yang sesuai dengan minat mereka [1][2]. Dari hal tersebut dapat dilihat bahwa dengan adanya sistem rekomendasi buku dapat memudahkan seseorang untuk mencari buku tanpa harus ragu apakah buku tersebut sesuai dengan minat mereka atau tidak. Sehingga dengan adanya proyek ini untuk memberikan rekomendasi buku apa yang sesuai dengan seseorang berdasarkan histori buku yang mereka baca dan berdasarkan pengalaman user lain dalam membaca sebuah buku. 

Dengan adanya sistem rekomendasi untuk memilih buku mana yang sesuai dengan karakteristik seseorang dapat mengurangi kesalahan seseorang dalam memilih buku. Selain itu, dengan adanya sistem rekomendasi buku diharapkan dapat meningkatkan minat membaca yang masih rendah khususnya di Indonesia [3]. Berdasarkan referensi nomor [3] sangat miris bahwa Indonesia menduduki peringkat ke-60 dari 61 negara soal minat membaca. Mengingat bahwa jumlah penduduk Indonesia yang sangat banyak seharusnya potensi membaca dapat lebih ditingkatkan kembali. Karena berdasarkan pepatah bahwa buku adalah jendela dunia. Buku merupakan sumber wawasan bagi manusia sehingga apabila kita memanfaatkannya dengan sebaik mungkin, secara tidak langsung dapat memajukan kesejahteraan masyarakat. Selanjutnya, sistem rekomendasi buku masih sangat jarang untuk ditemui, berbeda halnya dengan sistem rekomendasi pada film yang sudah sangat banyak di internet. 

## Business Understanding
Jika kita ada diposisi sebagai orang yang kebingungan untuk menentukan buku apa yang akan kita baca atau yang akan dibeli dipasaran. Maka kita membutuhkan sistem yang dapat merekomendasikan buku yang sekiranya sesuai dengan preferensi dan kebutuhan kita masing-masing, mengapa demikian? dikarenakan kita tidak ingin salah dalam membeli buku selain menghamburkan uang itu juga merupakan hal yang mubazir karena tidak bermanfaat. Dalam prakteknya banyak harga buku yang cukup mahal dipasaran apalagi jika kita mencari buku-buku internasional yang eksklusif atau sulit untuk ditemui dipasaran indonesia sehingga harga yang harus dibayarkan tidaklah sedikit. Sehingga dengan adanya sistem rekomendasi, user dapat mengetahui buku berdasarkan riwayat user itu sendiri dalam membaca buku dan berdasarkan pengalaman user lain dalam membaca sebuah buku berupa rating dari buku yang sudah dibaca. Kemudian data tersebut akan diolah oleh sistem rekomendasi untuk menghasilkan sistem rekomenadasi buku yang akurat. 

### Problem Statements
Berdasarkan _problem_ yang telah dijelaskan pada poin sebelumnya, saya akan mengembangkan sistem rekomendasi buku untuk menjawab permasalahan berikut.
- Terdapat beberapa atribut dalam menentukan sistem rekomendasi buku, atribut apa saja yang digunakan untuk membuat sistem rekomendasi buku?
- Bagaimana cara membuat model sistem rekomendasi buku menggunakan Metode Content Based Filtering dan Collaborative Filtering?  
- Bagaimana cara membuat model sistem rekomendasi buku menggunakan Machine Learning?

### Goals
Untuk  menjawab pertanyaan tersebut, saya akan membuat _recommendation system_ dengan tujuan atau _goals_ sebagai berikut:
- Mengetahui atribut apa saja yang digunakan dalam membuat model sistem rekomendasi buku.
- Mengetahui cara membuat model sistem rekomendasi buku menggunakan Metode Content Based Filtering dan Collaborative Filtering.
- Membuat model machine learning untuk sistem rekomendasi buku berdasarkan atribut yang ada.

### Solution statements
- Untuk mencapai goals yang sudah dijelaskan pada poin sebelumnya, yaitu sistem rekomendasi buku berdasarkan beberapa atribut yang berada pada buku itu sendiri dan dari _user experience_ dalam bentuk rating buku. Dalam membuat model sistem rekomendasi buku digunakan dua jenis metode yang berbeda yaitu _Content Based Filtering_ dan _Collaborative Filtering_. _Content Based Filtering_ merupakan metode yang dibuat berdasarkan atribut-atribut yang berada pada buku itu sendiri seperti, judul buku dan _publisher_. Sedangkan _Collaborative Filtering_ merupakan metode yang menggabungkan atribut pada buku dan _user experience_ dari yang sudah pernah membaca suatu buku dan memberikan rating terhadap sebuah buku sehingga akan menghasilkan _output_ berupa sistem rekomendasi buku yang mirip dengan buku yang sudah pernah dibaca oleh user tersebut. 
-  Metrik yang digunakan pada metode _Content Based Filtering_ adalah TF-IDF Vectorizer dan Cosine Similarity yang diperoleh dari matriks TF-IDF untuk menentukan _similarity_ seperti yang di contohkan pada modul dicoding. Kemudian pada metode _Collaborative Filtering_ menggunakan metrik Root Mean Squared Error (RMSE). Metrik RMSE bekerja dengan cara mengukur akurasi rata-rata antara pasangan individual pada nilai forecast dan Observasi.

## Data Understanding

Data yang saya gunakan pada proyek kali ini adalah _Books Recomender-Popular-Collaborative with WebApp_ yang diperoleh dari platform penyedia dataset Kaggle. Data tersebut berisikan _file_ Book.csv dan Ratings.csv, pada data buku berisikan beberapa atribut seperti ISBN, _Book Title_, _Book Author_, _Year_Of_Publication_, dan _Publisher_. Sedangkan pada _file_ rating berisikan atribut berupa _User-ID_, ISBN, _Book Rating_. Dataset tersebut diambil dari platform Kaggle dengan link berikut: [Kaggle](https://www.kaggle.com/code/methoomirza/books-recomender-popular-collaborative-with-webapp/data?select=Books.csv).
 
Dataset pada Book.csv terdiri dari 271360 data dan 5 Atribut. Sedangkan pada Ratings.csv terdiri dari 1048575 data dan 3 Atribut. Semua atribut yang berada pada dataset Book.csv tersebut bertipe object sedangkan pada dataset Ratings.scv bertipe 2 integer dan 1 object. 

### Variabel-variabel pada _Books Recomender-Popular-Collaborative with WebApp_ Dataset adalah sebagai berikut:
- File Book.csv berisikan:
  - ISBN : Kode dari suatu buku.
  - Book Title : Judul dari sebuah buku.
  - Book Author : Penulis dari sebuah buku.
  - Year Of Publication  : Tahun terbit dari sebuah buku.
  - Publisher : Penerbit dari sebuah buku.


- FIle Ratings.csv berisikan:
  -  User ID : Kode user yang memberikan rating terhadap sebuah buku.
  -  ISBN : Kode dari suatu buku.
  -  Book Rating : Rating yang diberikan user terhadap sebuah buku.

**Tahapan proses yang dilakukan pada proses Data Understanding**
- Tahap pertama adalah melakukan data loading dari dataset untuk mengetahui jumlah dataset yang dapat dilihat dari jumlah baris dan kolom yang terdapat pada dataset. Setelah dataset diload dilanjutkan dengan mengetahui shape dari dataset dengan memanggil fungsi .shape sehingga dapat diketahui jumlah baris dan kolom dari setiap file dataset yaitu Book.csv dan Ratings.csv sebanyak Books Shape (271360, 5) dan Ratings Shape (1048575, 3).
- Tahap selanjutnya adalah Univariate Exploratory Data Analysis yaitu mengeksplorasi masing-masing dataset menggunakan fungsi .info agar mengetahui tipe data pada masing-masing atribut. Semua atribut yang berada pada dataset Book.csv tersebut bertipe object sedangkan pada dataset Ratings.scv bertipe 2 integer dan 1 object. 
- Dilanjutkan dengan mengidentifikasi _missing value_ yang dari dataset  Book.csv dan Ratings.csv dengan cara memanggil fungsi .isnull().sum(). Diperoleh pada dataset Book.csv terdapat 1 missing value pada Book Author dan 2 missing value pada Publisher sedangkan pada dataset Ratings.csv tidak memiliki _missing value_. 
- Kemudian dilanjutkan dengan menghilangkan _missing value_ dengan cara memanggil fungsi .dropna() pada dataset Book.csv. Selanjutnya menghapus kolom yang tidak akan digunakan datanya menggunakan fungsi .drop() dari variabel data pada dataset Book.csv. Pada kasus ini kolom yang tidak akan digunakan adalah _'Year Of Publication'_ dan _'Book Author'_. Dikarenakan sistem rekomendasi buku akan menggunakan atribut _Book Title_, ISBN, dan _Publisher_. 
- Tahap terakhir pada Data Understanding adalah menyatukan atribut pada file dataset Ratings.csv dan Book.csv menjadi satu dataframe yang sama menggunakan fungsi pd.merge dari library pandas yang disimpan pada variabel all_book.   


## Data Preparation
Setelah melakukan Exploratory Data Analysis (EDA) dilanjutkan ke tahap Data Preparation. Dimana pada tahap ini data akan disiapkan yang nantinya akan diolah menggunakan metrik evaluasi TF-IDF pada metode _Content Based Filtering_ dan akan ditraining pada metode _Collaborative Filtering_. Dikarenakan dataset sudah digabungkan maka perlu memastikan kembali bahwa tidak ada _missing value_ yaitu memanggil fungsi .isna().sum() pada variabel all_book. Selanjutnya saya melakukan beberapa tahap data preparation yaitu: 
- Mengurangi jumlah dataset yang semula 83643 baris menjadi 500 baris.
- Menghapus data yang memiliki duplikat dengan fungsi memnaggil drop_duplicates. (Metode Content Based Filtering)
- Melakukan konversi dataseries menjadi list. (Metode Content Based Filtering)
- Membuat variabel dictionary sebelum melakukan pengolahan data. (Metode Content Based Filtering)
- Mempersiapkan data training pada model sistem rekomendasi pada metode Collaborative Filtering.

**Menjelaskan proses dan alasan mengapa diperlukan tahapan data preparation tersebut**:
- Tahap berikutnya adalah melakukan drop pada variabel book rate dikarenakan jumlah dataset yang terlalu besar sebanyak 83643 baris. Variabel book rate akan di drop pada range (500, 83643) sehingga dataset yang akan digunakan sebanyak 500 sampel data dan disimpan pada variabel book_new. Pengurangan jumlah dataset yang signifikan dikarenakan dengan jumlah sampel data sebanyak 83643 terlalu banyak yang menyebabkan google colab crash saat memasuki tahap TF-IDF dan juga acuan pada modul yang hanyak menggunakan sekitar 100 sampel data.
- Perlu penghapusan data jika terdapat data yang terduplikasi hal tersebut harus dilakukan agar satu data mewakili satu atribut. Penghapusan data ganda dilakukan pada variabel _all book_ diperoleh jumlah baris sebanyak 83643 yang semula sebanyak 941109. Ternyata banyak sekali data yang terduplikasi pada variabel tersebut. Dalam proses ini setiap satu user harus mewakili satu judul buku sehingga banyak data yang dihilangkan. Untuk menghapus data yang terduplikasi dapat memanggil fungsi drop_duplicates().
- Tahap selanjutnya adalah melakukan konversi data series menjadi list. Pada proses ini menggunakan fungsi tolist() dari library numpy. Mengapa data harus dikonversi? dikarenakan tidak semua data dapat diolah secara langsung. Oleh karena itu diperlukan adanya konversi data series menjadi data list agar data dapat diolah. 
- Selanjutnya adalah membuat _dictionary_ untuk menentukan pasangan key-value pada data userID,isbn, book rating, book title, dan book publisher yang telah disiapkan sebelumnya dan disimpan ke dalam variabel book rate. Dictionary dibutuhkan untuk menyimpan value/nilai pada setiap key-nya sehingga setiap value akan mempunyai kode yang unik.
- Untuk metode _Collaborative Filtering_, terdapat beberapa tahapan Data preparation yang berbeda dibandingkan metode _Content Based Filtering_. Pada tahap ini, fitur ‘id’ pada data akan disandikan (encode) ke dalam indeks integer. Sama halnya dengan fitur ‘id’, fitur 'isbn' pada data juga akan disandikan (encode) ke dalam indeks integer. hasil dari encode adalah dengan hasil seperti berikut. Tahap berikutnya adalah memetakan (mapping) id yang sudah di encode ke dalam dataframe user dan isbn ke dataframe book. Selanjutnya cek beberapa hal dalam data seperti jumlah user, jumlah buku, dan mengubah nilai rating menjadi float serta menghitung nilai min max dari atribut book_rating kemudian print hasilnya dan diperoleh nilai berikut Number of User: 500, Number of Book: 473, Min Rating: 0, Max Rating: 10. Tahap berikutnya adalah mengacak dataset sebelum dilakukan pembagian dataset  data training dan validasi. Dataset diacak agar sebaran data lebih random, digunakan parameter randomstate dengan nilai 42 dan parameter frac merupakan nilai float. Setelah dataset diacak dilanjutkan membagi dataset menjadi data train dan validasi dengan komposisi 90:10. Namun sebelum membagi, perlu ada pemetaan (mapping) data user dan book menjadi satu value yang sama dan juga membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.

## Modeling
Selanjutnya masuk ke tahap Model Development, pada tahap ini saya mengembangkan model machine learning yang bertujuan untuk menjawab problem statement dari tahap business understanding yaitu sistem rekomendasi buku. Pada tahap ini menggunakan 2 jenis Metode _Content Based Filtering_ dan _Collaborative Filtering_ seperti yang dicontohkan pada submodul Dicoding study kasus keempat : Sistem Rekomendasi.
 - Metode _Content Based Filtering_ 
   - Tahap pertama modeling adalah menggunakan teknik TF-IDF Vectorizer. Teknik tersebut akan digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap kategori buku. Skor dalam TF-IDF digunakan untuk mengamati istilah-istilah berbeda yang mengandung informasi penting dalam dokumen tertentu. Teknik ini menggunakan fungsi tfidfvectorizer() dari library sklearn.
   - Kemudian merubah bentuk fit dan transformasi ke dalam bentuk matriks, dapat dilihat ukuran matriks yang diperoleh adalah 500 x 293. Nilai 500 merupakan ukuran dataset dan 293 merupakan matriks dari atribut book_publisher. 
   - Untuk menghasilkan vektor tf-idf dalam bentuk matriks dapat menggunakan fungsi todense(). Objek matriks dengan bentuk yang sama dan berisi data yang sama yang diwakili oleh matriks sparse, dengan urutan memori yang diminta.
   - Tahap selanjutnya adalah membuat dataframe untuk melihat matriks tf-idf yang telah dibuat pada proses sebelumnya. Kolom diisi dengan publisher buku dan bari diisi dengan judul bukui. Data frame yang dihasilkan berjumlah 20 sampel acak judul buku, dan 10 sampel acak publisher.
   - Selanjutnya masuk ke tahap cosine similarity, pada tahap ini judul buku yang akan dihasilkan akan dihitung similaritynya berdasarkan matriks tfidf yang sudah dihasilkan sebelumnya menggunakan fungsi cosine_similarity(tf-idf matrix).
   - Sama seperti pada tf-idf, pada cosine similarity juga dibuat dataframe yang berisikan matriks hasil cosine similarity untuk membandingkan kemiripan antara judul satu buku dengan buku lainnya. Bentuk matriks cosine similarity adalah (500, 500) menandakan seluruh dataset sudah diidentifikasi menggunakan cosine similarity. Selanjutnya, ditampilkan sampel acak sebanyak 10 sampel pada baris dan kolom. Banyak judul buku yang tidak sesuai karena nilai yang diperoleh sebesar 0 dikarenakan data sampel diambil secara acak dan menyebabkan masih banyak judul buku yang tidak muncul.
   - Berikutnya adalah membuat fungsi book_recommendations, fungsi book_recommendations akan diambil k jumlah dengan nilai similarity terbesar pada index matrix yang diberikan (i). Fungsi book_recommendations memiliki beberapa parameter sebagai berikut:
       - nama_buku : nama buku berdasarkan index kemiripan dataframe.
       - similarity_data : Dataframe mengenai similarity yang telah diidentifikasi sebelumnya yaitu cosine_sim_df.
       - items : nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah book_title dan book_publisher.
       - k : Banyak rekomendasi yang ingin diberikan.
       
    - Tahap selanjutnya adalah mengambil satu sampel buku yang berada pada dataset dalam kasus ini berada di dalam variabel data. Saya mengambil contoh untuk memasukan judul buku "The Sigma Protocol".

|   	| id       | isbn    | book_rating | book_title        | book_publisher     |
|---	|----------|---------|-------------|-------------------|--------------------|
| 186 	| 	277367 |312276885| 10          |The Sigma Protocol | St._Martin's_Press |
    
   - - Tahap terakhir adalah menguji coba apakah sistem rekomendasi sudah berjalan dengan baik dengan cara memanggil fungsi book_recommendations dan mengisi parameter dengan judul buku yang ingin dicari similaritynya berdasarkan publisher buku tersebut. Diperoleh 5 judul buku yang memiliki kemiripin yang sama dengan buku "The Sigma Protocol" berdasarkan book_publishernya. Maka sistem rekomendasi menggunakan metode Content Based Filtering sudah berhasil dibuat dengan menghasilkan output berupa Top-N Recommendation seperti pada tabel di bawah.
   
   
|   	| book_title                                        | book_publisher     |
|---	|---------------------------------------------------|--------------------|
| 0 	|Trunk Music (Detective Harry Bosch Mysteries)	    | St._Martin's_Press |
| 1 	|The Far Pavilions                                  | St._Martin's_Press |
| 2 	|The Last Coyote (Last Coyote)	                    | St._Martin's_Press |
| 3 	|Biggie and the Mangled Mortician (Dead Letter ...  | St._Martin's_Press |
| 4 	|Ten Big Ones: A Stephanie Plum Novel               | St._Martin's_Press |


- Metode _Collaborative Filtering_
    - Metode _Collaborative Filtering_ memiliki perbedaan dengan metode sebelumnyayaitu membutuhkan interferensi dari data  user, dalam hal ini ada data book rating yang diberikan oleh user terhadap buku tertentu yang sudah pernah dibaca. Kemudian rating buku tersebut akan digunakan sebagai acuan terhadap buku-buku yang mirip dan belum pernah dibaca oleh user sehingga akan direkomendasikan. 
    - Untuk dataset yang digunakan masih sama seperti sebelumnya yaitu variabel book_new yang sekarang disimpan pada variabel df agar tidak menimpa variabel sebelumnya.
    - Untuk tahapan data preparation dan pre-processing sudah dijelaskan pada bagian sebelumnya.
    - Tahap selanjutnya adalah proses training, sebelum melakukan proses training dibuat class RecommenderNet dengan library keras. Model akan menghitung skor kecocokan antara user dan book dengan teknik embedding. Pada fungsi tersebut dilakukan proses embedding terhadap data user dan book. Setelah itu dilakukan operasi dot product antara user_vector dan book_vector, user_vector dan book_vector berisikan embedding dari user dan book itu sendiri. Nilai kecocokan yang diperoleh akan memiliki nilai dalam skala [0,1] dengan fungsi aktivasi sigmoid.
    - Kemudian melakukan proses compile pada model, model berisikan fungsi RecommenderNet yang sudah dideklarasikan sebelumnya. model.compile memiliki beberapa nilai parameter yaitu Binary Crossentropy untuk menghitung nilai loss function, kemudian Adam Adaptive Moment Estimation sebagai parameter optimizer dengan learning rate sebesar 0.0001, dan root mean squared error (RMSE) sebagai metrics evaluation yang digunakan pada model tersebut. 
    - Selanjutnya dilakukan training pada model dengan memanggil fungsi model.fit() yang didalamnya dideklarasikan beberapa parameter seperti data latih dari atribut dan label, batch_size yang digunakan sebesar 8, epoch sebesar 25, dan vaildation data pada variabel x_val y_val.
    - Tahap selanjutnya adalah memplot hasil training model ke dalam bentuk grafik dengan sumbu y sebagai nilai loss dan sumbu x sebagai banyaknya epoch seperti gambar di bawah ini. Dapat dilihat model yang dihasilkan sudah memiliki nilai loss yang cukup baik pada data train dan testnya.
    - Tahap selanjutnya untuk mendapatkan sistem rekomendasi buku adalah mendeklarasikan sampel user_id secara acak (dalam kasus ini saya mengambil sampel dari user id 276744) kemudian mendefinisikan variabel book_read_by_user yang merupakan daftar book yang sudah pernah dibaca oleh user tersebut. Kemudian mendeklarasikan variabel book_not_read merupakan daftar buku yang belum pernah dibaca oleh user tersebut. Variabel book_not_read nantinya akan menjadi acuan dari buku  yang direkomendasikan. 
    - Tahap terakhir untuk memperoleh hasil rekomendasi buku dengan N-Top recommendation seperti contoh pada modul adalah dengan cara mendeklarasikan model.predict() dimana model merupakan hasil training sebelumnya. Hasil dari prediksi akan menampilkan 10 jenis buku yang sesuai dengan user_id yang sudah dideklarasikan sebelumnya. Buku yang ditampilkan berdasarkan perkiraan rating tertinggi yang akan diberikan user_id terhadap buku yang belum pernah dibaca dikarenakan metode ini merupakan metode Collaborative Filtering dengan hasil seperti pada tabel di bawah. 
--------------------------------
Top 10 books recommendation
--------------------------------
|   	| isbn       | ratings    | id          | book_title                                        |
|-------|------------|------------|-------------|---------------------------------------------------|
| 0 	| 425173739  |0.548134    | 277782	    | Death on the Nile (Hercule Poirot Mysteries (P... | 
| 1 	| 671522728  |0.547028    | 277826      | Lucky, Lucky Day (Full House Michelle)            |   
| 2 	| 140118993  |0.546442    | 277830      | Tupelo Nights (Contemporary American Fiction)     | 
| 3 	| 3442729785 |0.544645    | 277457	    | Melancholie Der Ankunft                           | 
| 4 	| 786811358	 |0.544454	  | 277831	    | Jenius: The Amazing Guinea Pig (Hyperion Chapt... | 
| 5 	| 553148001  |0.544378    | 277506	    | The Clan of the Cave Bear : a novel               | 
| 6 	| 142001740	 |0.544192    | 277803      | The Secret Life of Bees                           | 
| 7 	| 142001740	 |0.544192    | 277958      | The Secret Life of Bees                           | 
| 8 	| 804107149  |0.544188    | 277832	    | Ordinary Love and Good Will                       | 
| 9 	| 440228441  |0.544188    | 277797	    | I Know What You Did Last Summer                   | 

**Kelebihan dan kekurangan kedua Metode**
- Kelebihan Algoritma Collaborative Filtering :
    1. Menghasilkan rekomendasi yang berkualitas baik.
    2. Sistem Rekomendasi dapat teteap berjalan dalam kondisi content sulit dianalisis.
    3. Menghasilkan serendipity item.
- Kekurangan Algoritma Collaborative Filtering :
    1. Tidak memuat konten dari barang yang direkomendasikan.
    2. Membutuhkan parameter rating, sehingga jika ada data baru maka sistem tidak akan merekomendasikannya. 
    3. Menghasilkan data yang kurang akurat ketika penilaian pada satu data terlalu sedikit dan akan menjadi salah persepsi.
- Kelebihan Algoritma Content Based Filtering :
    1. Model dapat menangkap minat khusus pengguna, dan dapat merekomendasikan item khusus.
    2. Model tidak memerlukan data mengenai user.
- Kekurangan Algoritma Content Based Filtering :
    1. Tidak mampu menentukan profil dari user baru.
    2. Model hanya dapat membuat rekomendasi berdasarkan minat pengguna yang ada. 

## Evaluation
   - Pada Metode _Content Based Filtering_ menggunakan metrik evaluasi _Cosine Similarity_ yang merupakan cara menghitung tingkat kemiripan yang disebut sebagai degree of similarity. Nilai pada metrik evaluasi _Cosine Similarity_ akan mengukur nilai cosinus antara dua vektor dalam kasus ini adalah membanding dua vektor dari book_title. Berdasarkan metrik evaluasi _Cosine Similarity_ akan membandingkan judul buku yang satu dengan yang lainnya berdasarkan kesamaan dibagian publishernya. Sehingga dapat dilihat pada bagian sebelumnya ketika saya memasukan judul buku "The Sigma Protocol" dari publisher "St._Martin's_Press", maka akan ditampilkan kelima buku lain dengan judul berbeda tetapi dari publisher yang sama.
   - Pada Metode _Collaborative Filtering_ menggunakan metrik evaluasi Root Mean Squared Error (RMSE) pada saat training model. RMSE merupakan salah satu metrik evaluasi yang biasanya digunakan untuk melakukan evaluasi pada model regresi linear dengan cara mengukur akurasi dari sebuah model. RMSE bekerja dengan cara mengkuadratkan nilai error kemudian hasil kuadrat tersebut dibagi dengan jumlah data dan selanjutnya diakarkan. Hal ini bertujuan agar nilai RMSE berada dalam skala yang tidak terlalu besar. Dari hasil training model dapat dilihat bahwa model memberikan nilai akhir RMSE sebesar 0.3815 dan loss 0.7459 didapat dari jumlah training sebanyak 25 epoch dengan batch size sebesar 8. Hasil tersebut sudah cukup baik dan dapat dilihat juga pada grafik dari model loss di bawah yang dihasilkan dari model yang sudah ditraining  menandakan model yang goodfit.
   ![Model Loss](https://github.com/mifbal99/test/blob/main/grafik%20training.png?raw=true) 

**Terdapat 3 jenis metrik yang digunakan yaitu:**
  - Metrik Cosine Similarity 
  Metrik _Cosine Similarity_ bekerja dengan cara  memberikan nilai dalam rentang [0, 1] terhadap dua vektor yang dibandingkan. Jika nilai dari kedua vektor tersebut mendekati maka akan mendekati nilai 1 sedangkan jika nilai kedua vektor tersebut berjauhan/tidak berhubungan maka akan mendekati nilai 0. Pengukuran tingkat kemiripan berkaitan dengan perhitungan jarak dari masing-masing atribut. Hubungan antara kemiripan dan jarak adalah berbanding terbalik, jika semakin besar nilai jarak maka kesamaan kemiripannya rendah dan sebaliknya. Selain itu metrik _cosine similarity_ mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama dengan cara menghitung sudut cosinus antara dua vektor tersebut. Semakin kecil nilai sudut, maka akan semakin besar nilai cosine similarity-nya.

  - Metrik Presisi
  Metrik Presisi adalah metrik evaluasi lain yang digunakan pada metode _Content Based Filtering_ selain _cosine similarity_. Pada sistem rekomendasi presisi merupakan jumlah item rekomendasi yang berhubungan. Untuk menggunakan metrik presisi dideklarasikan rumus yaitu P(presisi) = hasil rekomendasi yang relevan/jumlah total rekomendasi. Sebagai contoh, untuk mengetahui hubungan dari hasil sistem rekomendasi dapat dilihat dari kategori item yang direkomendasikan, dimana dalam kasus ini adalah publisher dari sebuah buku. Pada hasil yang sudah dibahas sebelumnya yaitu user pernah membaca buku "The Sigma Protocol" maka akan direkomendasikan buku yang memiliki kesamaan dengan buku dengan publisher yang sama dengan buku "The Sigma Protocol" yaitu St._Martin's_Press. Berdasarkan hasil pada tabel di bawah  didapat dari 5 rekomendasi buku tersebut ternyata seluruh buku relevan dengan buku yang pernah dibaca oleh user tersebut. Sehingga nilai presisinya menjadi 100% dikarenakan P = 5/5 x 100% = 100% maka hasil metrik presisi sebesar 100%.

|   	| book_title                                        | book_publisher     |
|---	|---------------------------------------------------|--------------------|
| 0 	|Trunk Music (Detective Harry Bosch Mysteries)	    | St._Martin's_Press |
| 1 	|The Far Pavilions                                  | St._Martin's_Press |
| 2 	|The Last Coyote (Last Coyote)	                    | St._Martin's_Press |
| 3 	|Biggie and the Mangled Mortician (Dead Letter ...  | St._Martin's_Press |
| 4 	|Ten Big Ones: A Stephanie Plum Novel               | St._Martin's_Press |

  - RMSE 
  Cara kerja metrik RMSE seperti yang sudah di _mention_ sebelumnya yaitu dengan cara mengkuadratkan nilai error kemudian hasil kuadrat tersebut dibagi dengan jumlah data dan selanjutnya diakarkan. Tujuan dari diakarkan adalah agar nilai rmse berada dalam skala yang tidak terlalu besar. Dari hasil training dapat dilihat bahwa model memberikan nilai akhir RMSE sebesar 0.3815 dan loss 0.7459 didapat dari jumlah training sebanyak 25 epoch dengan batch size sebesar 8. Hasil tersebut sudah cukup baik untuk sistem rekomendasi.

## Conclusion
Berdasarkan hasil dari kedua metode yaitu _Content Based Filtering_ dan _Collaborative Filtering_ yang digunakan untuk membuat sistem rekomendasi buku dapat disimpulkan, bahwa kedua metode dapat bekerja dengan baik dalam memprediksi buku dengan N-TOP Recommendation. Pada metode _Content Based Filtering_ sistem rekomendasi dapat menghasilkan 5 buku yang sesuai dengan _Publisher_ yang sesuai. Kemudian pada metode _Collaborative Filtering_, sistem rekomendasi dapat menghasilkan 10 urutan buku berdasarkan rating yang akan diberikan oleh user terkait. Sehingga model dari kedua metode sudah bagus dalam merekomendasikan buku. 

## Daftar Pustaka
- [1] Mathew, Praveena Kuriakose, , _Book Recommendation System through content based and collaborative filtering method_ , Proceedings of 2016 International Conference on Data Mining and Advanced Computing, SAPIENCE 2016, Issue 2016 | ISSN: 47-56.
- [2] Monique Ross. 2022, 11 Aug. _Best books for readers short on time, as recommended by Jennifer Down, Johann Hari, Tony Birch and more_. [Tautan Artikel](https://www.abc.net.au/news/2022-08-11/authors-recommend-best-books-for-time-poor-busy-readers/101291118)
- [3] Evita Devega. 2017, 10 Okt. _TEKNOLOGI Masyarakat Indonesia: Malas Baca Tapi Cerewet di Medsos_. [Tautan Artikel](https://www.kominfo.go.id/content/detail/10862/teknologi-masyarakat-indonesia-malas-baca-tapi-cerewet-di-medsos/0/sorotan_media#:~:text=Menurut%20data%20UNESCO%2C%20minat%20baca,1%20orang%20yang%20rajin%20membaca!)