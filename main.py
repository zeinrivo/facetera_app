import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import pickle
from matplotlib.patches import Rectangle

model_loaded = pickle.load(open('faceCounter_model','rb'))
uploaded_file = st.file_uploader("JPG only", type=["jpg"])

if uploaded_file is None:
    st.text("Please upload an Image")
else:
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    pixels = asarray (image)
    outcome = model_loaded.detect_faces(pixels)
    total = len(outcome)
    string = str(total)
    st.success(string)

    def draw_facebox(filename, result_list):
      data = plt.imread(filename)
      plt.imshow(data)
      ax = plt.gca()
      for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='yellow')
        ax.add_patch(rect)
      plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(draw_facebox(uploaded_file, outcome))
