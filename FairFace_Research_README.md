# Facial Race Classification Using a Custom CNN Architecture on the FairFace Dataset

## 1\. Introduction & Research Overview

This research implements and evaluates a custom Convolutional Neural Network (CNN) architecture for the task of facial race classification using the standard **FairFace** dataset. Leveraging modern regularization and optimization techniques, the proposed model provides stable performance in distinguishing and identifying various racial groups.

---

## 2\. Data Preprocessing & Management

To load and manage images, a custom `FairFaceDataset` class inheriting from `torch.utils.data.Dataset` was implemented, handling CSV label files, image path resolution, and mapping text labels to numeric indices.

The preprocessing and data augmentation pipeline includes:

* **Training Data (`train_transform`):** Resizing images to $56 \\times 56$ pixels, applying random horizontal flips (`RandomHorizontalFlip` with $p=0.5$), and converting to tensors.  
* **Validation Data (`val_transform`):** Resizing images to $56 \\times 56$ pixels and converting to tensors (without random data augmentation).

Data batches of size 64 are prepared using `DataLoader` for training and validation.

---

## 3\. Model Architecture (`RaceCNN`)

The proposed model is a sequential convolutional network comprising feature extraction blocks and classification layers:

1. **Convolutional Blocks:** Three consecutive blocks containing `Conv2d` layers (with 32, 64, and 128 filters respectively), batch normalization (`BatchNorm2d`), `ReLU` activation, and max-pooling (`MaxPool2d` with kernel size 2 and stride 2).  
2. **Dimensionality Transformation:** Flattening the output feature tensor to appropriate dimensions for fully connected layers ($128 \\times 7 \\times 7$).  
3. **Fully Connected Layers (Classifier):**  
   * First linear layer with an output of 256, followed by a `ReLU` activation and a `Dropout` layer with a rate of 0.5 to control overfitting.  
   * Final linear layer mapping to the total number of racial classes ($n\_class$).

---

## 4\. Training Configuration & Hyperparameters

The model was trained under the following configurations and hyperparameters:

* **Number of Epochs:** 15  
* **Batch Size:** 64  
* **Initial Learning Rate:** $1 \\times 10^{-4}$  
* **Weight Decay:** $1 \\times 10^{-4}$ (associated with the AdamW optimizer)  
* **Loss Function:** `CrossEntropyLoss`  
* **Optimizer:** `AdamW`  
* **Learning Rate Scheduler:** `ReduceLROnPlateau` (mode `max` on validation accuracy, factor 0.5, patience 2 epochs).

Throughout the training process, the best model weights based on the highest validation accuracy were saved to `best_model.pth`.

---

## 5\. Results & Evaluation

After completing 15 training epochs, the model achieved a **best validation accuracy of 59.38%**.

### Classification Report

The granular evaluation of the model on the validation set across 7 racial classes shows the following results:

* **Black:** Precision \= 0.75, Recall \= 0.79, F1-Score \= 0.77 (highest performance among classes)  
* **East Asian:** Precision \= 0.62, Recall \= 0.66, F1-Score \= 0.64  
* **Indian:** Precision \= 0.60, Recall \= 0.64, F1-Score \= 0.62  
* **Latino\_Hispanic:** Precision \= 0.44, Recall \= 0.43, F1-Score \= 0.44  
* **Middle Eastern:** Precision \= 0.49, Recall \= 0.42, F1-Score \= 0.45  
* **Southeast Asian:** Precision \= 0.58, Recall \= 0.46, F1-Score \= 0.51  
* **White:** Precision \= 0.62, Recall \= 0.68, F1-Score \= 0.65

The macro and weighted averages both indicate an overall accuracy of 0.59. Considering the challenges of facial images and small input dimensions ($56 \\times 56$), this basic structure provides a solid foundation for future research in facial feature analysis and deep learning. The Confusion Matrix further illustrates the model's error patterns in distinguishing various races.

&nbsp;