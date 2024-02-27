import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import transforms as T
import torch.optim as optim
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
from sklearn.metrics import recall_score, f1_score, precision_score
from tqdm import tqdm


ans = input('Did you update the paths for test_data, test_labels, and model_path? Type YES or NO.\n').upper()
if ans == 'YES':
    print("Running file...")
else:
    print("Exiting program...")
    exit()    
    
test_data_path = 'data_train.npy'
test_labels_path = 'labels_train.npy'
model_path = '/blue/eel5840/darian.jennings/final_proj/opt_model.pth'

test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)

# Define device to utilize and num_of_classes for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def test():
    # CREATE THE DATAFLOW FOR TEST
    # Standard IMAGENET vals sourced from google
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stds = [0.229, 0.224, 0.225]

    # Define AutoAugment transform 
    # - automatically augments data based on a given auto-augmentation policy - IMAGENET
    augmenter = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

    # Resize, augment,...
    # NOTE - In PyTorch, T.ToTensor() is a transformation that converts a PIL image or numpy array to a 
    # tensor and scales the values to the range [0.0, 1.0] 
    # NOTE - Normalize expects Tensor - so convert to tensor then normalize (*order matters*)

    # create a Tensor dataset (performs transformations AND augmentation)
    class TensorDataset(Dataset):    
        def __init__(self, data, labels, transform=None):
            self.data = data 
            self.labels = labels 
            self.transform = transform

        def __len__(self):
            return self.data.shape[1]

        def __getitem__(self, idx):
            image = self.data[:, idx].reshape(300, 300, 3)
            image = Image.fromarray(image) 
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx] 
            return image, label

    preprocess = T.Compose([
        augmenter,
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(imagenet_means, imagenet_stds)
    ])

    test_dataset = TensorDataset(test_data, test_labels, transform=preprocess)
    print("Created TEST -- Tensor Dataset")

    # Use DataLoader - load data into model - flexible for memory constraints
    test_dataflow = DataLoader(test_dataset, batch_size=256)
    
    nclasses = 9
    # Create instance of pre-trained VGG19 model
    def pretrainedVGG19(num_classes):
        model = vgg19(weights=VGG19_Weights.DEFAULT)           # pre-trained ImageNet weights
        num_ftrs = model.classifier[6].in_features             # extract n_ftrs in output layer
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) # n_output_features= n_classes
        return model
    
    # Call model instance (if not previously called) and load the saved model from path
    model = pretrainedVGG19(nclasses) 
    model.load_state_dict(torch.load(model_path))

    # Move the model on the same device as the data, either CPU or GPU, for the model to process data
    model = model.to(device) 
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_accuracy, test_loss = 0, 0
    correct, total = 0, 0
    unknown_indices = []
    test_losses = []
    test_accuracies = []

    print("---BEGIN TESTING---")
    with torch.no_grad():
        for data, target in test_dataflow:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            
            # calculate the probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            #max_prob, _ = torch.max(probabilities, dim=1)
            #print(max_prob)    
            # creates a boolean tensor unknown where each element is True if all 
            # probabilities in the corresponding row of the probabilities tensor are less than 0.5, 
            # and False otherwise
            unknown = (probabilities < 0.25).all(dim=1)
            _, predicted = torch.max(output.data, 1)
            #print(predicted)
            # classify unknown images to the unknown class (-1)
            predicted[unknown] = -1
            unknown_indices.extend(i for i, x in enumerate(predicted) if x == -1)
                
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1) 
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

    # Calculate test metrics
    test_loss /= len(test_dataflow.dataset) 
    test_accuracy = 100 * correct / total
    y_true = target.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    print(f'Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')
    print("---TEST COMPLETE---")
    print("Unknown indices: ", set(unknown_indices))


if __name__ == "__main__":
    test()

