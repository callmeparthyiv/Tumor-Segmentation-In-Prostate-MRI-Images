#importing libraries
import streamlit as st
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras.models import load_model

st.set_page_config(
    page_title="Prostate Tumor Segmentation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


#CSS
st.markdown(
    """
        <style>
        
            .main {
                background-color: #f5f5f5;
            }
            
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }
            
            .stFileUploader>div>div>div>button {
                background-color: #2196F3;
                color: white;
            }
            
            
            .title {
                font-size: 2.5em;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 20px;
            }
            
            .subtitle {
                    font-size: 1.5em;
                    color: #3498db;
                    margin-bottom: 15px;
            }
            .sidebar .sidebar-content {
                background-color: #2c3e50;
                color: white;
            }
            .result-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            }            
            
            
        </style>
    """, unsafe_allow_html=True
)

st.markdown('<h1 class = "title">Prostate Tumor Segmentation </h1>', unsafe_allow_html=True)

#preprocess the uploaded mri files
def preprocess(image_file, label = False):
    if image_file.name.endswith('.mhd'):
        image_file = '/content/drive/MyDrive/project/resized_preprocessed_dataset1/training_data/' + image_file.name
        img = sitk.ReadImage(image_file)
        img_array =  sitk.GetArrayFromImage(img)
        if label == False:
            img_array = scaler.fit_transform(img_array.reshape(-1, img_array.shape[0])).reshape(img_array.shape)
        return img_array
    else:
        return None
    
#plotting original mri image and original mask     
def plot_mri_slices(original, mask, slice_idx = 15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original MRI
    ax1.imshow(original[slice_idx,:,:], cmap='gray')
    ax1.set_title("Original MRI", fontsize=14)
    ax1.axis('off')
    
    # Original Segmentation
    ax2.imshow(mask[slice_idx,:,:], cmap='gray')
    ax2.set_title("Original Segmentation", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


#plotting original maska and predicted mask
def plot_prediction(mask, prediction, slice_idx = 15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original segmentation
    ax1.imshow(mask[slice_idx,:,:], cmap='gray')
    ax1.set_title("Original Segmentation", fontsize=14)
    ax1.axis('off')
    
    # Predicted Segmentation
    ax2.imshow(prediction[slice_idx,:,:], cmap='gray')
    ax2.set_title("Predicted Segmentation", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<p class="subtitle">Upload MRI Image </p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    #to upload mri and mask files
    with col1:
        image_file = st.file_uploader("Upload Prostate MRI Image", type=['mhd'])

    with col2:
        mask = st.file_uploader("Upload Ground Truth", type=['mhd'])
        
     

    if image_file and mask:
        
        mri_volume = preprocess(image_file) 
        mri_mask = preprocess(mask,label=True)
        st.pyplot(plot_mri_slices(mri_volume, mri_mask))
        
        
        if mri_volume is not None:
            mri_volume = np.expand_dims(mri_volume, axis = -1)
            
    #to process the input         
    process_btn = st.button("Process", type="primary")  
    
    #when button is pressed
    if process_btn:
        with st.spinner("Processing image..."):
            
            result_container = st.container()
            result_container.markdown('<p class="subtitle"> Results</p>', unsafe_allow_html=True)
            
            #prediction
            model = load_model('model/model.keras')
            mri_volume = np.expand_dims(mri_volume, axis=0)
            y_pred = model.predict(mri_volume)
            
            #show output
            result_container.pyplot(plot_prediction(mri_mask, y_pred[0],15))
        
if __name__ == "__main__":
    main()