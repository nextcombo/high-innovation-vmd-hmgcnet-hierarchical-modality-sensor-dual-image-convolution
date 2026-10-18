# High Innovation! VMD-HMGCNet: Hierarchical Modality - Sensor Dual Image Convolution Multi-source Vibration Fault Diagnosis Network

Preface
In the process of fault diagnosis for complex mechanical equipment, single sensors often only observe vibration responses at local positions of the equipment. Sensors with different measurement points and directions have different structural transmission paths and fault-sensitive frequency bands. At the same time, the original vibration signals usually exhibit obvious non-stationarity, modulation characteristics, periodic impacts, and band mixing phenomena. Directly concatenating multi-sensor signals into CNN, Transformer, or ordinary GCN can achieve end-to-end classification, but it often fails to explicitly distinguish: frequency band relationships, measurement point relationships, and raw domain-decomposed domain relationships. To address this, we propose a novel model for fault diagnosis of multi-source vibration signals based on variational mode decomposition: Variational Mode Decomposition Hierarchical Multi-Source Graph Convolution Fault Diagnosis Network (VMD-HMGCNet).
\tVMD-HMGCNet first utilizes VMD to decompose each sensor signal into multiple modal components with different frequency band attributes, and then constructs according to the true data hierarchy relationship: \tModal level graph > Sensor level graph\t, thereby separately learning:
\tDependence between different frequency modes within the same sensor;
Dynamic structural coupling relationship between different sensors;
Complementary information between the original vibration signal and VMD decomposition features.
\tThe final device fault status recognition is completed through hierarchical feature interaction.
\t(Suitable for signal classification tasks involving multiple sensors and features)
● Dataset: Southeast University Bearing Fault Dataset
● Environment Framework: Python 3.10, PyTorch 2.1 and above can be run.
● Accuracy: 100% on the test set.
● Usage Object: Paper requirements, graduation design requirements.
● Code Guarantee: The code comments are detailed and ready to run.
1. Overview of the Model Process
1.1 Overall Process of the Model
The core idea of MD-HMGCNet is to treat the different modalities after VMD decomposition as graph nodes with band-spectral semantics, and the different sensors as graph nodes with spatial source semantics. This is achieved through a hierarchical propagation from "modality-level graph to sensor-level graph".

The overall workflow of the model is as follows:
1. Branch one: Multi-scale encoding of raw signals [B,8,d]-> Cross-layer gate control fusion of raw signal and modality feature -> 8-node sensor adaptive graph -> Gate-controlled convolution -> Sensor-level attention convergence
2. Branch two: Raw 8-channel vibration signal [B, 1024, 8] -> Channel-wise VMD decomposition, each channel decomposed into 4 modalities -> Multi-source VMD modal sequence [B, 1024, 32] -> Multi-scale shared temporal node encoding -> 8-node modality graph within channels, each graph containing 4 modalities nodes -> Modality-level adaptive graph learning and multi-hop gate-controlled convolution -> Modality attention pooling [B,8,d]
3. Feature fusion final embedding -> Classifier -> Fault category
From an information fusion perspective, MD-HMGCNet models three types of relationships:
 (1)Modality internal relationship: The complementary relation between different VMD modalities of the same sensor.
 (2)Sensor-to-sensor relationship: The structural transmission and fault response coupling between different measurement points and directions.
 (3)Raw signal and decomposed signal relationship: The complementary relation between the instantaneous transient impact information and narrow-band modality information of VMD.
Therefore, MD-HMGCNet is not just a simple "VMD + GCN", but a multi-source graph fusion network organized by signal physical source and propagated information from modality to sensor level.
1.2 Multi-source feature signal VMD decomposition
For multi-source fault vibration signals, the Variational Mode Decomposition (VMD) is first applied to decompose the raw feature signal into multiple modalities. This process effectively separates complex signals into several Intrinsic Mode Functions (IMFs), capturing the local time-frequency characteristics of the signal. Additionally, VMD has strong noise suppression capabilities, providing a cleaner and more discriminative input for subsequent feature extraction.
1.3 Model Innovation Points:
Innovation Point One: Proposes a dual-layer structure for modalities and sensors. This is the core structural innovation of VMD-HMGCNet. The traditional VMD-GCN method often directly flattens: S×K modalities into: N=SK graph nodes. This approach mixes two different relationships into one graph: the cross-band relationship between the same sensor and the structural coupling relationship between different sensors; VMD-HMGCNet splits this into two levels.
The first level: Modality-level Graph, which means each sensor forms a modal summary that integrates multi-band information;
The second level: Sensor-level Graph, with a hierarchical structure that enables the model to first address "which frequency bands are important within a single measurement point" before solving "how to couple different measurement points." This is more clear in physical interpretation than constructing a large node graph directly.
Innovation Point Two: Dynamic—Global—Prior Mixed Graph Learning. VMD-HMGCNet does not use a fixed adjacency matrix. Whether it's the modality graph or the sensor graph, they are determined by three different sources:
Dynamic Graph + Global Graph + Prior Graph
Sample adaptive dynamic graph: Different fault states can have different modal relationship and sensor coupling structures;
Global learnable graph: In addition to current sample relationships, there may also be long-term stable node relationships within the device itself.
Signal Prior Graph: The third type of adjacent information comes from signal processing knowledge, modality prior + sensor relevance prior.
The three graphs are not manually weighted but define three learnable parameters: λ = [λd, λg, λp]. Therefore, the model can automatically decide which of the following is more relevant to the current task:
Innovation Point Three: Original Domain - VMD Decomposition Domain Cross-Layer Gating Fusion
Transient peak;
\tBroadband shock;
\tPhase relationship;
Original magnitude combination structure.
Therefore, VMD-HMGCNet does not adopt the "only input VMD component" scheme but instead employs feature dimension-level adaptive selection plus explicit multiplicative interactions.
Innovation Point Four: Modal Summary—Sensor Summary Double-Layer Semantic Interaction At the same time, the code does not discard the modal layer information but re-flattens the S×K modal nodes after modal propagation through global attention to obtain: m=Zmode. Therefore, two high-level semantics are ultimately retained. This design allows the final classification to not only rely on the last-level sensor graph but also retain discriminative information from the global modality band layer.
2. Preprocessing of multi-source bearing fault data
2.1 Import Data
Southeast University dataset is mainly divided into bearing dataset and gearbox dataset, with each fault type corresponding to these two working conditions (speed 20Hz (1200rpm) - load 0V(0Nm) and speed 30Hz (1800rpm) - load 2V(7.32Nm)), which are respectively represented by the grateset and bearingset folders. The bearing dataset includes five types:
1. Rolling Ball Fault (Ball Fault)
2. Inner Ring Fault (Inner Ring Fault)
3. Outer Ring Fault (Outer Ring Fault)
4. Combination Fault on Both Inner Ring and Outer Ring (Combined Fault on Both Inner Ring and Outer Ring)
5. Normal Working State (Healthy Working State)
We will preprocess the dataset with a 5-class bearing dataset, which is an 8-channel dataset!
2.2 VMD Decomposition and Preprocessing
Decomposition Visualization:
Due to the large volume of data, we take a part of the data as an example, create a dataset through sliding windows, and divide the dataset into training set, validation set, and test set in an equal ratio of 7:2:1, finally saving the data. The distribution of the dataset after completion is as follows:
3. Feature Fusion Model for Fault Diagnosis Based on VMD-HMGCNet
3.1 Define the Network Model
3.2 Set parameters and train the model
Trained for 50 epochs, accuracy 100%, using the VMD-HMGCNet feature fusion network classification effect is significant, it can extract important features from multi-source fault signal features for model recognition, the effect is very obvious!
3.3 Model Evaluation
(1) Accuracy, Precision, Recall, F1 Score
(2) Fault 5-Class Confusion Matrix:
(3) Classification Label Visualization
(4) Original Data t-SNE Feature Visualization
(5) Training After t-SNE Feature Visualization:
\tVMD-HMGCNet is a hierarchical dynamic graph network for multi-source vibration fault diagnosis. It achieves structured modeling and interpretable diagnostics of complex multi-measurement point fault information through VMD band prior, two-level modal-sensor graph propagation, three-source topology learning of dynamic-global-prior, multi-hop gate-controlled graph convolution, and original domain-decomposed domain cross-layer fusion.
Click to download: Full text data, Python code
Previously interesting content:
Python Bear Fault Diagnosis (19) Innovative Diagnosis Model Based on Transformer-BiLSTM
END

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
