import streamlit as st
import time
import matplotlib.pyplot as plt
from MTL_RUN import pipeline, depth_to_rgb
import numpy as np
from PIL import Image
import tempfile
import cv2

if "photo" not in st.session_state:
    st.session_state["photo"]="not done"

st.title("HydraNets (Segmentation and Depth)")

st.sidebar.title("Operations")

st.markdown("""

    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        background-color: black;
    }
    body {background-color: powderblue;}

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
unsafe_allow_html=True,)

# st.sidebar.markdown('---')
# confidence = st.sidebar.slider('Confidence Bar', min_value=0.0, max_value=1.0, value=0.25)
# st.sidebar.markdown('---')

# save_img = st.sidebar.checkbox('Save Video')
# enable_GPU = st.sidebar.checkbox('enable GPU')
# segm = st.sidebar.checkbox('Segm')
# depth = st.sidebar.checkbox('depth')


col1, col2, col3 = st.columns([2,2,2])
col1.markdown("test")
col2.markdown("test2")

def change_photo_state():
    st.session_state["photo"]="done"

uploaded_photo = st.sidebar.file_uploader("Upload a photo", on_change=change_photo_state)
# uploaded_video = st.sidebar.file_uploader("Upload a video",accept_multiple_files=True, on_change=change_photo_state)
# camera_photo = col2.camera_input("take a photo", on_change=change_photo_state)


def load_image(image_file):
    img = Image.open(image_file)
    return img


img = np.array(Image.open(uploaded_photo))

depth, segm = pipeline(img)


depth_rgb = depth_to_rgb(depth)

new_img = np.vstack((img,segm, depth_rgb))



if st.session_state["photo"]=="done":
    progress_bar = col2.progress(0)

    for perc_completed in range(20):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)
    col2.success("Photo upload successfully")

    col3.metric(label="Temperature", value="60 c", delta="3 c")
    # st.image(new_img)

    fig, ax = plt.subplots()
    im = ax.imshow(new_img)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 20))
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=30)
    ax2.imshow(segm)
    ax2.set_title('Predicted Segmentation', fontsize=30)
    ax3.imshow(depth, cmap="plasma", vmin=0, vmax=80)
    ax3.set_title("Predicted Depth", fontsize=30)

    st.pyplot(fig)

    # result_video = []
    # for idx, img_path in enumerate(uploaded_video):
    #     image = np.array(Image.open(img_path))
    #     h, w, _ = image.shape
    #     depth, seg = pipeline(image)
    #     result_video.append(cv2.cvtColor(cv2.vconcat([image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))
    #
    # # st.video(result_video)
    # out = cv2.VideoWriter('output/out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (w, 3 * h))
    # for i in range(len(result_video)):
    #     out.write(result_video[i])
    # out.release()
    # st.video(out)
    with st.expander("Click to read more"):
        st.write("Hello, here are more details on this topic that you were interested in.")
        if uploaded_photo is None:
            pass

            # st.image(camera_photo)
        else:
            st.image(uploaded_photo)
