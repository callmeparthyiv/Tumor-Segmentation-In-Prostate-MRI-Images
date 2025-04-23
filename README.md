# Tumor Segmentation in Prostate MRI Using UNet
Streamlit application to detect and segment tumors in 3D MRI prostate images

## About
There are various techniques in practice to detect prostate tumors such as CT scans or MRI. The problem with these techniques is that they are extremely tedious and require expert supervision, causing it to be a time-consuming process. This project focuses on the segmentation of prostate tumors from 3D MRI image volumes using deep learning. Modified 3D Unet architecture has been designed and used for the project. The model was trained and saved to use the model for application. The application has been deployed using localtunnel on Google Colab.

## Requirements
```markdown
- Python 3.10
- Numpy 2.1.3
- Matplotlib 3.10.0
- Scikit-Learn 1.6.1
- SimpleITK 2.4.1
- Tensorflow 2.18.0
- Streamlit 1.44.0
```

### **Usage**
Explanation on how to run the project

```markdown
1. Firstly you would need an environment with tensorflow 2.18 and GPU support
2. Install all the required items from requirements
3. Run the app.py file
4. Upload the MRI file using the interface
5. Click process & that's it.
```


