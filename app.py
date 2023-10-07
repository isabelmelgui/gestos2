import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
#Mi NuavaApp
import os
import time
import glob
from gtts import gTTS
from PIL import Image
#imagen
import pytesseract


model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Lector de etiquetas")
st.title('_Moda_ :violet[inclusiva]')


image = Image.open('Moda1.jpg')
st.image(image, width= 700)

img_file_buffer = st.camera_input("Toma una Foto de tu etiqueta")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('puño, con Probabilidad: '+str( prediction[0][0]))
      text = "puño, con Probabilidad El vestido es rojo, tiene un corte clásico, con una longitud que llega justo por encima de la rodilla, lo que lo hace fácil de llevar en diversas ocasiones, ya sea una cena formal o una reunión casual. La parte delantera presenta un escote redondo que se siente suave al tacto y es fácil de encontrar. Además, los botones en la parte posterior del vestido están diseñados de manera táctil, lo que permite abrocharlos sin dificultad."
    if prediction[0][1]>0.5:
      st.header('palma, con Probabilidad: '+str( prediction[0][1]))
      text = "Palma, con Probabilidad, Este chaleco de color gris es una prenda versátil y elegante que puede agregar un toque de sofisticación a cualquier conjunto. El diseño de este chaleco es clásico y atemporal, con un corte ajustado que realza la figura de quien lo lleva puesto. Tiene un cierre de botones frontales que facilita su colocación y ajuste. Los botones están cuidadosamente cosidos en un patrón que agrega un toque de estilo y sofisticación."


#Mi NuevaApp


try:
    os.mkdir("temp")
except:
    pass

st.subheader("Escucha las características de la prenda")


#text = st.text_input("Ingrese el texto.")

tld="es"

def text_to_speech(text, tld):
    
    tts = gTTS(text,"es", tld, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text


#display_output_text = st.checkbox("Verifica el texto")

if st.button("Escuchar características"):
    result, output_text = text_to_speech(text, tld)
    audio_file = open(f"temp/{result}.mp3", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Tú audio:")
    st.audio(audio_bytes, format="audio/mp3", start_time=0)

    #if display_output_text:
    st.markdown(f"## Texto en audio:")
    st.write(f" {output_text}")


def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)


remove_files(7)

image2 = Image.open('Moda2.jpg')
st.image(image2, width= 700)

#Leer imagen

st.title("También puedes cargar la imagen para ser leida")

img_file_buffer2 = st.file_uploader("Choose a jpg file")

with st.sidebar:
      filtro2 = st.radio("Aplicar Filtro",('Con Filtro', 'Sin Filtro'))


if img_file_buffer2 is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer2.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    if filtro2 == 'Con Filtro':
         cv2_img =cv2.bitwise_not(cv2_img)
    else:
         cv2_img = cv_img
    
        
    img_rgb2 = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    text=pytesseract.image_to_string(img_rgb2)
    st.write(text) 
