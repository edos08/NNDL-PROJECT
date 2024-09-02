# Cars Classification Project

This repository contains the principal code for a cars classification project, primarily implemented in the `cars_classification.ipynb` Jupyter Notebook. The additional support files, namely `models.py`, `utils.py` and `dataset.py`, provide supplementary functions and model architectures for the main code.

## Project Structure

- **cars_classification.ipynb:** The main Jupyter Notebook file containing the principal code for cars classification.

- **moco_supcon_cars_classification.py:** The main Python script containing the principal code for pre-training using MoCo SupCon.

- **lin_supcon_cars_classification.py:** The main Python script containing the main code for for cars classification using the pretrained network from MoCo SupCon.

- **test_model.ipynb:** Jupyter Notebook file containing the principal code testing the models.

- **models/resnet.py:** A Python script containing ResNet architecture and other supporting files used in the project.

- **models/moco_supcon.py:** A Python script containing MoCo SupCon architecture and other supporting files used in the project.

- **others/utils.py:** A Python script with utility functions and data processing tools to support the main code.

- **others/dataset.py:** A Python script with a custom dataset class for loading.

- **others/losses.py:** A Python script implementing Supervised Contrastive Loss function used for MoCo SupCon.

- **others/print_results.ipynb:** A Jupyter Notebook used to print intermediate results.

- **others/cars_classification.py:** A Python script containing a script version of the Jupyter Notebook.

- **trained_models:** This folder contains pre-trained model weights saved in .pth files. These files store the learned parameters of various neural network architectures. (Now contains a link to download the pre-trained models).

- **requirements.txt:** A file listing the necessary libraries and their versions. To install the required dependencies, run:
  ```
  pip install -r requirements.txt
  ```

## Getting Started

1. Open and run the `cars_classification.ipynb` notebook using Jupyter or any compatible environment.
2. The `models.py` and `utils.py` files are automatically imported by the main notebook to provide necessary functions and custom model architectures.
3. The `trained_models` directory includes pre-trained weights for various models, facilitating easy experimentation.

Feel free to explore the notebook for detailed insights into the weather classification project.
