# Classification of Breast Tumors Using Machine Learning 🔬

### 🎓 Diploma Project - Lăcătuş Denisa-Maria

---

## 📖 About The Project

This repository contains the code and findings for the diploma project, **"Classification of Breast Tumors Using Machine Learning"**. The project provides a comprehensive exploration into applying machine learning for the critical task of breast tumor classification.

> The core objective is to conduct a comparative analysis between a traditional machine learning model and a deep learning architecture to understand how their performance is influenced by the nature and volume of the input data.

Two distinct approaches were implemented and evaluated:
- A feature-based **Logistic Regression** model.
- An image-based **Convolutional Neural Network (CNN)**.

The study aims to provide a clear guide for selecting the optimal method based on the data available in an oncological context, moving from theory to practical implementation and evaluation.

---

## 🛠️ Methodology

### Datasets Used

#### Logistic Regression Dataset
- **Source**: The "Breast Cancer Wisconsin (Diagnostic) Dataset" was utilized.
- **Content**: The dataset comprises **569 samples** of breast nodules obtained through fine-needle aspiration (FNA).
- **Features**: For each sample, **30 numerical features** were computed, describing the morphological properties of the cell nuclei. These features are derived from 10 base characteristics (radius, texture, perimeter, etc.), with the mean, standard error (SE), and "worst" (largest value) calculated for each.
- **Labels**: Each sample is labeled as either benign (coded as `0`) or malignant (coded as `1`). The dataset is imbalanced, containing 357 benign and 212 malignant tumors.

#### Convolutional Neural Network (CNN) Dataset
- **Source**: The "Breast Tumor Mammography Dataset for Computer Vision" was used.
- **Content**: This dataset contains **3,383 grayscale mammography images**.
- **Resolution**: The original images have a resolution of $640 \times 640$ pixels.
- **Structure**: The data is pre-organized into `train` (2,371 files), `valid` (675 files), and `test` (336 files) directories, with subfolders for the benign (`0`) and malignant (`1`) classes.

### Data Preprocessing and Feature Engineering

- **For Logistic Regression**: The process involved cleaning the data of irrelevant columns (e.g., `ID`), applying **Min-Max Normalization** to scale all features to a common range of [0, 1], and calculating class weights to counteract the data imbalance during training.
- **For CNN**: Images were uniformly resized to **$224 \times 224$ pixels** to standardize the input size for the network. Pixel values were normalized from the [0, 255] range to [0, 1]. An efficient `tf.data` pipeline was built using `cache()` and `prefetch()` to optimize data loading and minimize GPU idle time.

### Development Environment & Tech Stack

The project was developed entirely within **Google Colab**, leveraging its free access to hardware accelerators (GPUs).

| Technology | Purpose |
| :--- | :--- |
| **`Python`** | Core programming language. |
| **`NumPy`** | Numerical operations and vectorization. |
| **`Pandas`** | Data loading, cleaning, and manipulation. |
| **`Scikit-learn`** | Data splitting, evaluation metrics, and PCA. |
| **`TensorFlow/Keras`**| Building, training, and evaluating the CNN model. |
| **`Matplotlib & Seaborn`**| Data visualization, learning curves, and confusion matrices. |

---

## ⚙️ Model Implementation

### Logistic Regression Model

The model was implemented from scratch using `NumPy` to provide a deep understanding of its mechanics.

- **Cost Function**: A custom function was implemented to calculate the binary cross-entropy loss, which included a **weighted L2 regularization** term to prevent overfitting.
- **Optimization**: A **Gradient Descent** algorithm was implemented to minimize the cost function. This algorithm was adapted to incorporate the pre-calculated class weights, ensuring that the model did not become biased towards the majority class.
- **Hyperparameters**: After multiple experiments, the final configuration used a learning rate ($\alpha$) of **0.008**, a regularization parameter ($\lambda$) of **0.01**, and was trained for **20,000 iterations**.

### Convolutional Neural Network (CNN) Model

A sequential model was designed using the Keras API to classify the mammography images.

- **Architecture**:
    - An input layer accepting `$224 \times 224 \times 1$` images.
    - **Three convolutional blocks**, each consisting of a `Conv2D` layer (`ReLU` activation) with 32, 64, and 128 filters respectively, followed by a `MaxPool2D` layer.
    - A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    - **Two `Dense` layers** with 128 and 64 neurons (`ReLU` activation), each followed by a `Dropout` layer (rate of 0.5) for regularization.
    - A final `Dense` output layer with a single neuron and a `Sigmoid` activation function.
- **Compilation & Training**: The model was compiled using the **AdamW optimizer** (learning rate of 1e-4) and `BinaryCrossentropy` loss. Training was controlled by `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (patience=3) callbacks to prevent overfitting.

---

## 📊 Results and Performance Evaluation

The two models yielded starkly contrasting results, highlighting the importance of matching model complexity to the data's characteristics.

### Logistic Regression

> **Excellent Performance on Structured Data**

- **Overall Accuracy**: The model achieved an outstanding accuracy of **98.25%** on the test set.
- **Confusion Matrix**: The evaluation on the test set of 114 samples resulted in:
    - **True Negatives (Benign)**: 71
    - **False Positives**: 1
    - **False Negatives**: 1
    - **True Positives (Malignant)**: 41
- **Metrics**: The model showed highly balanced performance with a precision and recall of **98.61% for the benign class** and **97.62% for the malignant class**. This indicates a highly reliable classifier with minimal errors.

### Convolutional Neural Network (CNN)

> **Underwhelming Performance Due to Data Limitations**

- **Overall Accuracy**: The CNN achieved a modest accuracy of **64%** on the test set.
- **Key Issue**: The model suffered from **severe overfitting**. While training accuracy steadily increased to 74%, validation accuracy stagnated around 67% before declining. The divergence between the training and validation loss curves confirmed this issue.
- **Confusion Matrix**: The model's primary weakness was its inability to correctly identify malignant cases:
    - **True Negatives (Benign)**: 194
    - **False Positives**: 14
    - **False Negatives**: 107
    - **True Positives (Malignant)**: 21
- **Metrics**: The **recall for the malignant class was only 16%**. This means the model failed to detect approximately **83% of the actual malignant tumors**, making it clinically unreliable.

---

## 🏁 Discussion and Future Work

The experimental results lead to a clear conclusion: for datasets with a limited number of samples and well-engineered features, a simple, interpretable model like **Logistic Regression can significantly outperform a complex deep learning model**. The 30 numerical features provided a strong, clear signal that the linear model could easily learn. In contrast, the CNN, with its 11 million trainable parameters, did not have enough diverse image data to learn generalizable features and instead memorized the training set.

This project highlights key challenges in applying deep learning in medicine, including the need for massive, diverse datasets and addressing the "black box" nature of complex models.

Future work to improve the CNN's performance could include:
- **Expanding the Dataset**: Acquiring tens of thousands of images from multiple medical centers to increase diversity and reduce bias.
- **Transfer Learning**: Using a pre-trained model (e.g., VGG, ResNet) and fine-tuning it on the medical images to leverage knowledge learned from larger datasets.
- **Advanced Data Augmentation**: Employing techniques like Generative Adversarial Networks (GANs) to create realistic synthetic tumor images for training.
