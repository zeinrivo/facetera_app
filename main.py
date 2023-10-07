import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
from mtcnn import MTCNN
from matplotlib.patches import Rectangle

st.set_page_config(
    page_title="FACETERA",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Home","Model","Demo"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)

if selected == "Home":
    st.write("# FACETERA ")
    st.write(
    """
    A Face Detection Web-Based App Using **Multi-Task Cascaded Convolutional Neural Networks (MTCNN)**.
    """
    )

    image2 = Image.open('home.jpg')
    image2.thumbnail((400,400))
    st.image(image2)
    
    st.markdown(
    """
    - [Source Code](https://github.com/zeinrivo/facetera_app)
    """
    )

    st.caption("Created by **Zein Rivo**")

if selected == "Model":
    st.write("# FACETERA ")
    st.write(
    """
    **Multi-Task Cascaded Convolutional Neural Networks (MTCNN)**.
    """
    )

    image1 = Image.open('model1.jpg')
    image1.thumbnail((800,800))
    image3 = Image.open('model2.jpg')
    image3.thumbnail((800,800))
    st.image(image1)
    st.image(image3)
    
    st.markdown(
    """
    - [Pypi](https://pypi.org/project/mtcnn/)
    """
    )
    st.markdown(
    """
    - [Google Colaboratory](https://drive.google.com/drive/folders/1IsV7ZTiEyOao3RoPiY-TympCGeiKOkPU?usp=share_link)
    """
    )


if selected == "Demo":
    model = MTCNN()
    uploaded_file = st.file_uploader("", type=["jpg"])
    if uploaded_file is None:
      st.text("Please upload an Image (jpg format only)")
    else:
      image = Image.open(uploaded_file)
      image = image.convert('RGB')
      pixels = asarray (image)
      outcome = model.detect_faces(pixels)
      total = len(outcome)
      string = str(total)
      st.success(string)

      def draw_facebox(filename, result_list):
        # load the image
        data = plt.imread(filename)
        # plot the image
        plt.imshow(data)
        # get the context for drawing boxes
        ax = plt.gca()
        # plot each box
        for result in result_list:
          # get coordinates
          x, y, width, height = result['box']
          # create the shape
          rect = plt.Rectangle((x, y), width, height, fill=False, color='yellow')
          # draw the box
          ax.add_patch(rect)
          # show the plot
        plt.show()

      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.pyplot(draw_facebox(uploaded_file, outcome))
