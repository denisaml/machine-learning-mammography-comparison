# Classification of Breast Tumors Using Machine Learning
### Diploma Project - Lăcătuş Denisa-Maria

---

## 📖 About The Project

[cite_start]This project provides a comprehensive exploration into the application of machine learning for the critical task of breast tumor classification[cite: 35]. [cite_start]The core objective is to conduct a comparative analysis between a traditional machine learning model and a deep learning architecture to understand how their performance is influenced by the nature and volume of the input data[cite: 36].

### Project Motivation
[cite_start]Traditional methods for diagnosing breast tumors, which rely on the expertise of radiologists and pathologists, can be influenced by subjective factors, potentially leading to diagnostic errors[cite: 32]. [cite_start]Machine Learning offers a powerful alternative, capable of rigorously and rapidly analyzing large volumes of medical data[cite: 33]. [cite_start]This project investigates two distinct approaches to automate this classification task: a feature-based **Logistic Regression** model and an image-based **Convolutional Neural Network (CNN)**[cite: 35].

[cite_start]The study aims to provide a clear guide for selecting the optimal method based on the data available in an oncological context, moving from theory to practical implementation and evaluation[cite: 62].

---

## 🛠️ Methodology

### Logistic Regression Dataset
* [cite_start]**Source**: The "Breast Cancer Wisconsin (Diagnostic) Dataset" was utilized[cite: 764].
* [cite_start]**Content**: The dataset comprises **569 samples** of breast nodules obtained through fine-needle aspiration (FNA)[cite: 764].
* [cite_start]**Features**: For each sample, **30 numerical features** were computed, describing the morphological properties of the cell nuclei[cite: 765]. [cite_start]These features are derived from 10 base characteristics (radius, texture, perimeter, etc.), with the mean, standard error (SE), and "worst" (largest value) calculated for each, resulting in 30 distinct inputs[cite: 774, 775, 776, 777].
* [cite_start]**Labels**: Each sample is labeled as either benign ('B', coded as 0) or malignant ('M', coded as 1)[cite: 765, 771]. [cite_start]The dataset is imbalanced, containing 357 benign and 212 malignant tumors[cite: 773].

### Convolutional Neural Network (CNN) Dataset
* [cite_start]**Source**: The "Breast Tumor Mammography Dataset for Computer Vision" was used[cite: 795].
* [cite_start]**Content**: This dataset contains **3,383 grayscale mammography images**[cite: 795, 49].
* [cite_start]**Resolution**: The original images have a resolution of $640 \times 640$ pixels[cite: 795].
* [cite_start]**Structure**: The data is pre-organized into `train` (2,371 files), `valid` (675 files), and `test` (336 files) directories, with subfolders for the benign (0) and malignant (1) classes[cite: 796, 1466].

### Data Preprocessing and Feature Engineering
* [cite_start]**For Logistic Regression**: The process involved cleaning the data of irrelevant columns (e.g., ID), applying **Min-Max Normalization** to scale all features to a common range [0, 1] [cite: 48, 241][cite_start], and calculating class weights to counteract the data imbalance during training[cite: 48, 981, 982].
* [cite_start]**For CNN**: Images were uniformly resized to **$224 \times 224$ pixels** to standardize the input size for the network[cite: 813]. [cite_start]Pixel values were normalized from the [0, 255] range to [0, 1] by dividing by 255.0[cite: 1469, 1472]. [cite_start]An efficient `tf.data` pipeline was built using `cache()` and `prefetch()` to optimize data loading and minimize GPU idle time[cite: 50, 1477].

### Development Environment
[cite_start]The project was developed entirely within **Google Colab**, leveraging its free access to hardware accelerators (GPUs)[cite: 820]. The key libraries used include:
* [cite_start]**NumPy & Pandas**: For numerical operations and data manipulation[cite: 822].
* [cite_start]**Scikit-learn**: For data splitting (`train_test_split`), evaluation metrics (`classification_report`, `confusion_matrix`), and dimensionality reduction (`PCA`)[cite: 825, 875, 876].
* [cite_start]**TensorFlow with Keras API**: For building, training, and evaluating the CNN model[cite: 826].
* [cite_start]**Matplotlib & Seaborn**: For creating visualizations, including learning curves, confusion matrices, and feature distributions[cite: 834].

---

## ⚙️ Model Implementation

### Logistic Regression Model
[cite_start]The model was implemented from scratch in Python using NumPy to provide a deep understanding of its mechanics[cite: 53].
* [cite_start]**Cost Function**: A custom function was implemented to calculate the binary cross-entropy loss, which included a **weighted L2 regularization** term to prevent overfitting[cite: 53, 1217].
* [cite_start]**Optimization**: A **Gradient Descent** algorithm was implemented to minimize the cost function[cite: 53]. [cite_start]This algorithm was adapted to incorporate the pre-calculated class weights, ensuring that the model did not become biased towards the majority class[cite: 53, 1239].
* **Hyperparameters**: Multiple training runs were performed to find the optimal hyperparameters. [cite_start]The final configuration used a learning rate ($\alpha$) of **0.008**, a regularization parameter ($\lambda$) of **0.01**, and **20,000 iterations**[cite: 1687].

### Convolutional Neural Network (CNN) Model
[cite_start]A sequential model was designed using the Keras API to classify the mammography images[cite: 55].
* **Architecture**:
    * [cite_start]An input layer accepting $224 \times 224 \times 1$ images[cite: 1505].
    * [cite_start]**Three convolutional blocks**, each consisting of a `Conv2D` layer (with 32, 64, and 128 filters respectively, and a $3 \times 3$ kernel) with a `ReLU` activation function, followed by a `MaxPool2D` layer for downsampling[cite: 55, 1506, 1508, 1510].
    * [cite_start]A `Flatten` layer to convert the 2D feature maps into a 1D vector[cite: 1512].
    * [cite_start]**Two `Dense` (fully connected) layers** with 128 and 64 neurons (`ReLU` activation), each followed by a `Dropout` layer (rate of 0.5) for regularization[cite: 55, 1513, 1516].
    * [cite_start]A final `Dense` output layer with a single neuron and a `Sigmoid` activation function to produce a probability score between 0 and 1[cite: 55, 1518].
* [cite_start]**Compilation & Training**: The model was compiled using the **AdamW optimizer** with a learning rate of 1e-4 and weight decay of 1e-4[cite: 55]. [cite_start]`BinaryCrossentropy` was used as the loss function[cite: 1588]. [cite_start]Training was set for 50 epochs with a batch size of 32, controlled by `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (patience=3) callbacks to prevent overfitting and adjust the learning rate dynamically[cite: 55].

---

## 📊 Results and Performance Evaluation

The two models yielded starkly contrasting results, highlighting the importance of matching the model complexity to the data's characteristics.

### Logistic Regression
* [cite_start]**Overall Accuracy**: The model achieved an outstanding accuracy of **98.25%** on the test set[cite: 1717, 1721].
* **Confusion Matrix**: The evaluation on the test set of 114 samples resulted in:
    * **True Negatives (Benign)**: 71
    * **False Positives**: 1
    * **False Negatives**: 1
    * [cite_start]**True Positives (Malignant)**: 41 [cite: 1743, 1744]
* [cite_start]**Metrics**: The model showed excellent and balanced performance with a precision and recall of **98.61% for the benign class** and **97.62% for the malignant class**[cite: 1728]. This indicates a highly reliable classifier with minimal errors.

### Convolutional Neural Network (CNN)
* [cite_start]**Overall Accuracy**: The CNN achieved a modest accuracy of **65%** on the test set[cite: 55, 1825, 1858].
* **Key Issue**: The model suffered from **severe overfitting**. [cite_start]While training accuracy steadily increased, validation accuracy stagnated and then declined[cite: 1977]. [cite_start]The divergence between the training loss (decreasing) and validation loss (stagnating/increasing) confirmed this issue[cite: 1979].
* **Confusion Matrix**: The model's primary weakness was its inability to correctly identify malignant cases:
    * **True Negatives (Benign)**: 193
    * **False Positives**: 15
    * **False Negatives**: 102
    * [cite_start]**True Positives (Malignant)**: 26 [cite: 1900, 1901, 1906, 1907]
* [cite_start]**Metrics**: The **recall for the malignant class was only 21%**[cite: 55, 1854, 1875]. [cite_start]This means the model failed to detect approximately **80% of the actual malignant tumors**, classifying them as benign[cite: 55]. Such a high false-negative rate would make the model clinically unreliable.

---

## 🏁 Discussion and Future Work

[cite_start]The experimental results lead to a clear conclusion: for datasets with a limited number of samples and well-engineered features, a simple, interpretable model like **Logistic Regression can significantly outperform a complex deep learning model**[cite: 57]. [cite_start]The 30 numerical features in the Wisconsin dataset provided a strong, clear signal that the linear model could easily learn [cite: 2010][cite_start], whereas the CNN, with its 11 million trainable parameters, did not have enough diverse image data to learn generalizable features and instead memorized the training set[cite: 2008, 57].

[cite_start]This project highlights the challenges of applying deep learning in medicine, including the need for massive, diverse datasets and the "black box" nature of complex models[cite: 2031, 2033].

Future work to improve the CNN's performance could include:
* [cite_start]**Expanding the Dataset**: Acquiring tens of thousands of images from multiple medical centers to increase diversity and reduce bias[cite: 2043].
* [cite_start]**Transfer Learning**: Using a pre-trained model (e.g., VGG, ResNet) and fine-tuning it on the medical images to leverage knowledge learned from larger datasets[cite: 58, 2044].
* [cite_start]**Advanced Data Augmentation**: Employing techniques like Generative Adversarial Networks (GANs) to create realistic synthetic tumor images for training[cite: 58, 2045].
