# X-ray Image Classifier

This is a Streamlit-based application that allows users to upload chest X-ray images and classify them as either "NORMAL" or "PNEUMONIA" based on a pre-trained deep learning model.

## Features

- Single image upload and classification
- Batch upload of multiple images for bulk processing
- Visualization of the input image with the predicted class and confidence score
- Probability distribution for the predicted classes

## Dataset Link

```bash
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

## Prerequisites

- Python 3.7 or later
- The following Python libraries:
  - TensorFlow
  - Numpy
  - Streamlit
  - Pillow
  - Matplotlib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sathiesh05/Classification_of_chest_Xray_by_CNN.git
```

2. Navigate to the project directory:

```bash
cd Classification_of_chest_Xray_by_CNN
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the pre-trained model file (`final.h5`) and place it in the project directory.

## Usage

To run the application, execute the following command:

```bash
python -m streamlit run app.py
```

This will launch the Streamlit application in your default web browser.

The application has two tabs:

1. **Single Image**: Upload a single X-ray image to get the classification prediction.
2. **Batch Processing**: Upload multiple X-ray images to process them in a batch.

The application will display the input image, the predicted class, the confidence score, and a probability distribution for the predicted classes.

## Deployment

To deploy the application, you can use a platform like Streamlit Sharing or host it on a cloud platform like AWS, Google Cloud, or Heroku.

## Contributing

If you find any issues or would like to contribute to the project, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/sathiesh05/Classification_of_chest_Xray_by_CNN).
