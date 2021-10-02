from utils import *

model_selection = ['Basic 10 CNN layer Serial Model', 'Fine tuned VGG16 Model', 'Densenet Model', 'Ensemble Model','Show all']

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("background.jpeg")
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write("""
         # Test Deepfake
         """
         )
st.write("Deep Fake Detector")
path = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])

st.write(path)
if path is None:
    st.text("Please upload an image file")
else:
    path = np.asarray(bytearray(path.read()), dtype=np.uint8)
    display_image(path)
    selection = st.selectbox('Model Selection:', model_selection)
    serial = serial(path)
    vgg16 = vgg16(path)
    mobilenet = mobilenet(path)
    densenet = densenet(path)
    out_string_e = cal_ensemble(serial,vgg16,densenet,"Ensemble   ")
    out_string_serial = generate_string(serial,"Serial     ")
    out_string_vgg16 = generate_string(vgg16,"VGG16      ")
    out_string_densenet = generate_string(densenet,"Densenet   ")


    if selection == model_selection[0]:
        st.write(out_string_serial)
    elif selection == model_selection[1]:
        st.write(out_string_vgg16)
    elif  selection == model_selection[2]:
        st.write(out_string_densenet)z
    elif  selection == model_selection[3]:
        st.write(out_string_e)
    elif selection == model_selection[4]:
        st.write(out_string_serial)
        st.write(out_string_vgg16)
        st.write(out_string_densenet)
        st.write(out_string_e)


        




    