import tf_keras as keras  # Mengimpor tf-keras – versi Keras yang kompatibel dengan model .h5
from tf_keras.models import load_model  # Mengimpor fungsi load_model dari tf_keras, yang memungkinkan kita mengakses modelnya
from PIL import Image, ImageOps  # Memasang pillow sebagai ganti PIL
import numpy as np
import h5py

#check the size of image <2MB
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# Menonaktifkan notasi ilmiah untuk kejelasan
np.set_printoptions(suppress=True)

f = h5py.File("keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")

if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
f.attrs.modify('model_config', model_config_string)
f.flush()

model_config_string = f.attrs.get("model_config")

assert model_config_string.find('"groups": 1,') == -1
# Memuat model
model = load_model("keras_model.h5", compile=False, safe_mode=False)

# Memuat label
class_names = open("labels.txt", "r").readlines()

# Membuat susunan bentuk yang tepat untuk dimasukkan ke dalam model keras
# 'Panjang', atau jumlah gambar yang dapat Anda masukkan ke dalam array adalah
# ditentukan oleh posisi pertama dalam tupel bentuk (dalam hal ini, 1)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Ganti ini dengan jalur ke gambar Anda
image = Image.open("/content/burungkecil.jpg")
image = image.convert("RGB")

# Mengubah ukuran gambar menjadi setidaknya 224x224px dan kemudian memotongnya dari bagian tengah
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Mengubah gambar menjadi array NumPy
image_array = np.asarray(image)

# Normalisasi gambarnya
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Memuat gambar ke dalam array
data[0] = normalized_image_array

# Memprediksi model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Mencetak prediksi dan skor keyakinan
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

def detect_bird(image, model, labels):
  np.set_printoptions(suppress=True)
  model = load_model(model, compile=False)
  class_names = open(labels, "r").readlines()

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  image = image.convert("RGB")

  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  image_array = np.asarray(image)

  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  data[0] = normalized_image_array

  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  return class_name[2:], confidence_score

if class_name[2:] == 'Organik\n':
  print("Sampah organik adalah limbah yang berasal dari sisa makhluk hidup (tumbuhan, hewan, manusia) yang dapat terurai secara alami oleh mikroorganisme. Sampah ini mudah membusuk dan umumnya mengandung air, seperti sisa makanan, daun kering, dan kotoran ternak. Sampah organik sering dimanfaatkan sebagai bahan kompos, pupuk, atau biogas. ")
elif class_name[2:] == 'Anorganik\n':
  print("Sampah anorganik adalah jenis limbah yang berasal dari bahan non-hayati, produk sintetik, atau hasil proses teknologi pengolahan bahan tambang yang umumnya sulit atau tidak dapat terurai secara alami oleh mikroorganisme. Sampah ini membutuhkan waktu sangat lama untuk terurai, seperti plastik, kaca, logam, kaleng, dan sterofoam. ")
