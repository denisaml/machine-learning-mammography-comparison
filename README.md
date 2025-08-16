# Classification of Breast Tumors Using Machine Learning 🔬

### 🎓 Diploma Project - Lăcătuş Denisa-Maria

---

## 📖 About The Project

This repository contains the code and findings for the diploma project, **"Classification of Breast Tumors Using Machine Learning"**. The project provides a comprehensive exploration into applying machine learning for the critical task of breast tumor classification.

> [cite_start]The core objective is to conduct a comparative analysis between a traditional machine learning model and a deep learning architecture to understand how their performance is influenced by the nature and volume of the input data[cite: 36].

Two distinct approaches were implemented and evaluated:
- [cite_start]A feature-based **Logistic Regression** model[cite: 35].
- [cite_start]An image-based **Convolutional Neural Network (CNN)**[cite: 35].

[cite_start]The study aims to provide a clear guide for selecting the optimal method based on the data available in an oncological context, moving from theory to practical implementation and evaluation[cite: 62].

---

## 🛠️ Methodology

### Datasets Used

#### Logistic Regression Dataset
- [cite_start]**Source**: The "Breast Cancer Wisconsin (Diagnostic) Dataset" was utilized[cite: 764].
- [cite_start]**Content**: The dataset comprises **569 samples** of breast nodules obtained through fine-needle aspiration (FNA)[cite: 764].
- [cite_start]**Features**: For each sample, **30 numerical features** were computed, describing the morphological properties of the cell nuclei[cite: 765]. [cite_start]These features are derived from 10 base characteristics (radius, texture, perimeter, etc.), with the mean, standard error (SE), and "worst" (largest value) calculated for each[cite: 774, 775, 776, 777].
- [cite_start]**Labels**: Each sample is labeled as either benign (coded as `0`) or malignant (coded as `1`)[cite: 765, 771]. [cite_start]The dataset is imbalanced, containing 357 benign and 212 malignant tumors[cite: 773].

#### Convolutional Neural Network (CNN) Dataset
- [cite_start]**Source**: The "Breast Tumor Mammography Dataset for Computer Vision" was used[cite: 795].
- [cite_start]**Content**: This dataset contains **3,383 grayscale mammography images**[cite: 49, 795].
- [cite_start]**Resolution**: The original images have a resolution of $640 \times 640$ pixels[cite: 49, 795].
- [cite_start]**Structure**: The data is pre-organized into `train` (2,371 files), `valid` (675 files), and `test` (336 files) directories, with subfolders for the benign (`0`) and malignant (`1`) classes[cite: 796, 1466].

### Data Preprocessing and Feature Engineering

- [cite_start]**For Logistic Regression**: The process involved cleaning the data of irrelevant columns (e.g., `ID`), applying **Min-Max Normalization** to scale all features to a common range of [0, 1], and calculating class weights to counteract the data imbalance during training[cite: 48, 805, 807].
- [cite_start]**For CNN**: Images were uniformly resized to **$224 \times 224$ pixels** to standardize the input size for the network[cite: 813]. [cite_start]Pixel values were normalized from the [0, 255] range to [0, 1][cite: 815]. [cite_start]An efficient `tf.data` pipeline was built using `cache()` and `prefetch()` to optimize data loading and minimize GPU idle time[cite: 50].

### Development Environment & Tech Stack

[cite_start]The project was developed entirely within **Google Colab**, leveraging its free access to hardware accelerators (GPUs)[cite: 820].

| Technology | Purpose |
| :--- | :--- |
| **`Python`** | Core programming language. |
| **`NumPy`** | [cite_start]Numerical operations and vectorization[cite: 822]. |
| **`Pandas`** | [cite_start]Data loading, cleaning, and manipulation[cite: 822, 824]. |
| **`Scikit-learn`** | [cite_start]Data splitting, evaluation metrics, and PCA[cite: 825]. |
| **`TensorFlow/Keras`**| [cite_start]Building, training, and evaluating the CNN model[cite: 826]. |
| **`Matplotlib & Seaborn`**| [cite_start]Data visualization, learning curves, and confusion matrices[cite: 834]. |

---

## ⚙️ Model Implementation

### Logistic Regression Model

The model was implemented from scratch using `NumPy` to provide a deep understanding of its mechanics.

- [cite_start]**Cost Function**: A custom function was implemented to calculate the binary cross-entropy loss, which included a **weighted L2 regularization** term to prevent overfitting[cite: 53].
- [cite_start]**Optimization**: A **Gradient Descent** algorithm was implemented to minimize the cost function[cite: 53]. [cite_start]This algorithm was adapted to incorporate the pre-calculated class weights, ensuring that the model did not become biased towards the majority class[cite: 53].
- [cite_start]**Hyperparameters**: After multiple experiments, the final configuration used a learning rate ($\alpha$) of **0.008**, a regularization parameter ($\lambda$) of **0.01**, and was trained for **20,000 iterations**[cite: 1687].

### Convolutional Neural Network (CNN) Model

A sequential model was designed using the Keras API to classify the mammography images.

- **Architecture**:
    - An input layer accepting `$224 \times 224 \times 1$` images.
    - [cite_start]**Three convolutional blocks**, each consisting of a `Conv2D` layer (`ReLU` activation) with 32, 64, and 128 filters respectively, followed by a `MaxPool2D` layer[cite: 55].
    - A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    - [cite_start]**Two `Dense` layers** with 128 and 64 neurons (`ReLU` activation), each followed by a `Dropout` layer (rate of 0.5) for regularization[cite: 55].
    - [cite_start]A final `Dense` output layer with a single neuron and a `Sigmoid` activation function[cite: 55].
- [cite_start]**Compilation & Training**: The model was compiled using the **AdamW optimizer** (learning rate of 1e-4) and `BinaryCrossentropy` loss[cite: 55]. [cite_start]Training was controlled by `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (patience=3) callbacks to prevent overfitting[cite: 55].

---

## 📊 Results and Performance Evaluation

The two models yielded starkly contrasting results, highlighting the importance of matching model complexity to the data's characteristics.

### Logistic Regression

> **Excellent Performance on Structured Data**

- [cite_start]**Overall Accuracy**: The model achieved an outstanding accuracy of **98.25%** on the test set[cite: 1721].
- [cite_start]**Confusion Matrix**: The evaluation on the test set of 114 samples resulted in[cite: 1743, 1744, 1752]:
    - **True Negatives (Benign)**: 71
    - **False Positives**: 1
    - **False Negatives**: 1
    - **True Positives (Malignant)**: 41
- [cite_start]**Metrics**: The model showed highly balanced performance with a precision and recall of **98.61% for the benign class** and **97.62% for the malignant class**[cite: 1748, 1749]. This indicates a highly reliable classifier with minimal errors.

### Convolutional Neural Network (CNN)

> **Underwhelming Performance Due to Data Limitations**

- [cite_start]**Overall Accuracy**: The CNN achieved a modest accuracy of **65%** on the test set[cite: 2004, 2018].
- [cite_start]**Key Issue**: The model suffered from **severe overfitting**[cite: 55]. [cite_start]While training accuracy steadily increased to 74%, validation accuracy stagnated around 67% before declining[cite: 1976, 1977]. [cite_start]The divergence between the training and validation loss curves confirmed this issue[cite: 1979].
- [cite_start]**Confusion Matrix**: The model's primary weakness was its inability to correctly identify malignant cases[cite: 1915]:
    - **True Negatives (Benign)**: 193
    - **False Positives**: 15
    - **False Negatives**: 102
    - **True Positives (Malignant)**: 26
- [cite_start]**Metrics**: The **recall for the malignant class was only 21%**[cite: 1875, 2018]. [cite_start]This means the model failed to detect approximately **80% of the actual malignant tumors**, making it clinically unreliable[cite: 55, 2020].

---

## 🏁 Discussion and Future Work

[cite_start]The experimental results lead to a clear conclusion: for datasets with a limited number of samples and well-engineered features, a simple, interpretable model like **Logistic Regression can significantly outperform a complex deep learning model**[cite: 57]. [cite_start]The 30 numerical features provided a strong, clear signal that the linear model could easily learn[cite: 2011]. [cite_start]In contrast, the CNN, with its 11 million trainable parameters, did not have enough diverse image data to learn generalizable features and instead memorized the training set[cite: 1577, 2009].

[cite_start]This project highlights key challenges in applying deep learning in medicine, including the need for massive, diverse datasets and addressing the "black box" nature of complex models[cite: 2033].

Future work to improve the CNN's performance could include:
- [cite_start]**Expanding the Dataset**: Acquiring tens of thousands of images from multiple medical centers to increase diversity and reduce bias[cite: 2031, 2043].
- [cite_start]**Transfer Learning**: Using a pre-trained model (e.g., VGG, ResNet) and fine-tuning it on the medical images to leverage knowledge learned from larger datasets[cite: 58, 2044].
- [cite_start]**Advanced Data Augmentation**: Employing techniques like Generative Adversarial Networks (GANs) to create realistic synthetic tumor images for training[cite: 2045].
