# 🩺 HC4AI – Explainability Module (Grad-CAM & Shapley)

This repository contains the **explainability part** of the HC4AI project,  
developed for the course *Human-Computer Interaction for Artificial Intelligence* (A.Y. 2022/2023, [Politecnico di Milano](https://www.polimi.it/)).

The complete project, including the conversational agent, can be found in the [colleague’s repo](https://github.com/rtjk/hci4ai-project).  
Here we focus on the **training and interpretation** of the CNN model for pneumonia detection.

---

## 📖 Overview

The project implements a convolutional neural network (CNN) for pneumonia detection on chest X-ray images,  
with a focus on **interpretability**:

- **Grad-CAM** → generates heatmaps showing the areas of the image most relevant for the model’s decision.  
- **Shapley values** → highlights pixel contributions (positive/negative) to improve model transparency.  

This approach aims to provide medical students with an interpretable tool to understand and learn from the model predictions.

---

## 🗂 Repository Structure

```
📁 data/                # Sample chest X-ray images
📁 data_preparation/    # Pre-training of the CNN model
📁 notebook/            # Scripts for Grad-CAM (hc4ai_gradcam.ipynb) and Shapley (hc4ai_shapley.ipynb)
📁 output/              # Generated outputs (heatmaps, Shapley plots)
📁 src/                 # Source code (.py)
📁 results/             # Final report (PDF) and demo video
📄 environment.yml      # .yaml to show conda environment spec
📄 README.md
```

---

## 📊 Scripts

- `hc4ai_gradcam.py` → Implementation of **Grad-CAM** to visualize pretrained CNN attention. 
- `hc4ai_gradcam_xception.py` → Implementation of **Grad-CAM** to visualize Xception (from Keras) attention.  
- `hc4ai_shapley.py` → Use of **SHAP** library to compute Shapley values for local explanations.  

Outputs are automatically saved under `output/`.

---

## 📑 Results

- 📄 **HC4AI.pdf** → Final project report.  
- 🎥 **demo.mp4** → Short demo showcasing explainability results.  
- 🖼️ Grad-CAM and Shapley plots available in the `output/` folder.

---

## ⚠️ Notes
 
- The CNN model is pre-trained and saved in `data_preparation/model_output/`.  
- Due to size limitations, full datasets are not included.  

---

## 👥 Authors

- Marco Gianvecchio  
- Giacomo Da Re  
- Lorenzo Manoni
