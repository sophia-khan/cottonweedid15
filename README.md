
# 🌱 Cotton Weed Species Identification with CNNs

This project focuses on automated weed identification in cotton fields using advanced deep learning architectures. By leveraging CNN models such as VGG16 (with soft attention), MobileNet, and ResNet50, we aim to enhance weed detection accuracy and reduce the labor-intensive demands of manual weed classification.

## 📌 Project Objective

To develop and evaluate CNN-based weed detection systems capable of identifying 15 common weed species in cotton production systems in the southern U.S., using RGB image data from the **CottonWeedID15** dataset.

---

## 🧠 Models Used

* **VGG16** with Custom Classifier and Soft Attention
* **MobileNet** with Depthwise Separable Convolutions
* **ResNet50** with Custom Classification Head

---

## 🧪 Dataset

* Source: [CottonWeedID15](https://www.kaggle.com/datasets/yuzhenlu/cottonweedid15)
* 5187 RGB images across 15 weed classes
* Split: 65% training / 20% validation / 15% test
* Preprocessing: Image resizing, normalization to \[-1, 1], data augmentation (rotation, flip)

---

## 🛠️ Features

* 📷 **Image Input**: Upload your weed image via GUI
* 🧠 **Model Selection**: Choose from VGG16 (w/soft attention), MobileNet, or ResNet50
* ✅ **Real-time Classification**: Output the identified weed class immediately

---

## 🖥️ GUI Demo

🎬 **Watch the GUI in action:**

https://github.com/user-attachments/assets/4405c4ea-2e4c-4aa2-82a0-f613318428b1


---

## 📊 Evaluation Metrics

All models were trained on the **TensorFlow** framework using **Google Colab** and **Kaggle**. Accuracy was used as the primary performance metric.

---

## ✅ Results

* High-performing classification system with real-time GUI
* Soft attention boosted interpretability and accuracy
* MobileNet and ResNet50 offered strong performance with smaller computational needs

---

## 🚀 Future Work

* Expand model generalization to other crops
* Integrate with real-time field tools (drones, IoT sensors)
* Further optimize accuracy and GUI responsiveness

---

## 🙏 Acknowledgements

This research was conducted at **Fort Valley State University** under the mentorship of **Dr. Xiangyan Zeng** and **Dr. Chunhua Dong**. Funded by the **NSF HBCU-UP** program and supported by **Google**.

---

## 📚 References

* Chen et al. (2022). *Performance evaluation of deep transfer learning on multi-class identification of common weed species in cotton production systems.* Computers and Electronics in Agriculture.
* Datta et al. (2021). *Soft-Attention Improves Skin Cancer Classification Performance.* medRxiv.

