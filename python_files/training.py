import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from transformers import AutoModel
import pandas as pd
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns

from .setup import load_DNA_model
from .preprocess_data import create_label_encoder
from .models import DNABertModel, CustomClassifier, ProteinOnlyClassifier, DNAOnlyClassifier, DualHeadProteinOnlyClassifier
from .datasets import ProteinDataset, DNADataset, CLSEmbeddingsDataset, ProteinAverageEmbeddingsDataset, ProteinMaxEmbeddingsDataset, CombinedDataset

def create_weighted_sampler_dataloader(dataset, label_encoder, full_label_set, balance_other=False, batch_size=16, shuffle=False):
    """
    Creates a weighted random sampler to handle class imbalance by sampling with replacement based on class frequencies.
    Optionally adjusts the "Other" class to make up exactly 50% of the data which balancing the remaining classes if 
    balance_other is True.
    """
    # Encode labels to their corresponding class indices
    encoded_labels = label_encoder.transform(full_label_set)

    # Count the occurrences of each class
    class_counts = np.bincount(encoded_labels)
    num_classes = len(class_counts)
    
    # Calculate weights as the inverse of class frequencies
    if balance_other:
        other_index = label_encoder.transform(['Other'])[0]
        total_samples = len(encoded_labels)
        desired_other_count = total_samples / 2  # 50% of the total data

        # Adjust weights for 'Other' to represent 50% of the data
        other_class_weight = desired_other_count / class_counts[other_index]

        # Adjust weights for classes other than 'Other'
        remaining_classes = num_classes - 1
        remaining_samples = total_samples - desired_other_count  # Remaining 50% of the data
        mean_weight_per_class = remaining_samples / remaining_classes  # Evenly distribute the remaining 50% among the other classes

        # Define the class weights
        class_weights = np.ones(num_classes) * (mean_weight_per_class / class_counts)
        class_weights[other_index] = other_class_weight
    else:
        class_weights = 1. / class_counts
    
    # Assign a weight to each sample based on its class
    sample_weights = class_weights[encoded_labels]

    # Create a sampler that samples with replacement based on these weights
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Setup the dataloader object
    dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    
    return dataloader


def predict_for_protein_only(model, data, device):
    '''Runs a prediction on a model that includes only protein information (no DNA information)'''
    protein_embeddings = data['embeddings'].to(device)

    # Forward pass
    prediction = model(protein_embeddings)
    return prediction

def predict_for_DNA_only(model, data, device):
    '''Runs a prediction on a model that includes only DNA information (no protein information)'''
    dna_embeddings = data['chunk_embeddings'].to(device)
    dna_attention_mask = data['attention_mask'].to(device)

    # Forward pass
    prediction = model(dna_embeddings, dna_attention_mask)
    return prediction


def predict_for_protein_and_DNA(model, data, device):
    '''Runs a prediction on a model that includes both protein information and DNA information'''
    dna_embeddings = data['dna_chunk_embeddings'].to(device)
    dna_attention_mask = data['dna_attention_mask'].to(device)
    protein_embeddings = data['protein_embeddings'].to(device)

    # Forward pass
    prediction = model(dna_embeddings, dna_attention_mask, protein_embeddings)
    return prediction

def run_inference_on_data(model, dataloader, label_encoder, device, model_type='', verbose=False):
    '''Takes a model, dataloader, and encoder, and runs inference through all data in the dataloader.
    Prints out a sample of the data and returns the accuracy, labels, and predictions'''
    accuracy = []
    all_labels = []
    all_predictions = []

    for i, data in enumerate(dataloader):
        
        #Run the prediction
        if model_type == 'DNA_only': prediction = predict_for_DNA_only(model, data, device)
        elif model_type == 'protein_only': prediction = predict_for_protein_only(model, data, device)
        elif model_type == 'protein_and_DNA': prediction = predict_for_protein_and_DNA(model, data, device)
        else: raise ValueError(f"Invalid model_type provided: {model_type}. Expected 'DNA_only', 'protein_only', or 'protein_and_DNA'.")

        # Collect labels
        labels = data['label']
        encoded_labels = label_encoder.transform(labels)
        labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

        # Calculate accuracy
        _, predicted_labels = torch.max(prediction, axis=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        accuracy.append(correct_predictions / len(labels))

        # Collect predictions and labels
        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted_labels.tolist())

        if verbose:
          print(f'Batch labels are: {labels}')

    print('predictions of last batch are:')
    print(prediction)
    print('-'*50)
    print(f'Accuracy is: {np.mean(accuracy)}')
    return accuracy, all_labels, all_predictions

def run_inference_on_data_multihead(model, dataloader, label_encoder, device, model_type='', verbose=False):
    '''Takes a model (specifically the multi-head classifier model), dataloader, and encoder, and runs inference 
    through all data in the dataloader. Prints out a sample of the data and returns the accuracy, labels, 
    and predictions for both classification heads'''
    binary_accuracy = []
    main_class_accuracy=[]
    all_binary_labels = []
    all_main_class_labels = []
    all_binary_predictions = []
    all_main_class_predictions = []

    for i, data in enumerate(dataloader):
        protein_embeddings = data['embeddings'].to(device)
        labels = data['label']
        binary_labels, main_class_labels = transform_labels(labels, label_encoder, device)

        #Run the prediction
        # Forward pass
        binary_probs, main_class_probs = model(protein_embeddings)

        # Mask for main class labels that are not 'Other'
        valid_main_class_mask = main_class_labels != -1

        # Calculate accuracy
        predicted_binary_labels = torch.where(binary_probs > 0.5, torch.ones_like(binary_probs), torch.zeros_like(binary_probs)).squeeze()
        _, predicted_main_class_labels = torch.max(main_class_probs[valid_main_class_mask], axis=1)
        correct_binary_predictions = (predicted_binary_labels == binary_labels).sum().item()
        correct_main_class_predictions = (predicted_main_class_labels == main_class_labels[valid_main_class_mask]).sum().item()
        binary_accuracy.append(correct_binary_predictions / len(labels))
        main_class_accuracy.append(correct_main_class_predictions / valid_main_class_mask.sum().item())

        # Collect predictions and labels
        all_binary_labels.extend(binary_labels.tolist())
        all_main_class_labels.extend(main_class_labels[valid_main_class_mask].tolist())
        all_binary_predictions.extend(predicted_binary_labels.tolist())
        all_main_class_predictions.extend(predicted_main_class_labels.tolist())

        if verbose:
          print(f'Batch labels are: {labels}')
    
    # Calculate overall accuracy:
    combined_accuracy = calculate_multihead_accuracy(all_binary_labels, all_main_class_labels, all_binary_predictions, all_main_class_predictions)

    print('predictions of last batch are:')
    print('binary probabilities:')
    print(binary_probs)
    print('-'*50)
    print('main class probs:')
    print(main_class_probs)
    print(f'Binary accuracy is: {np.mean(binary_accuracy)} \nMain class accuracy is: {np.mean(main_class_accuracy)}')
    print(f'Overall accuracy is: {combined_accuracy}')
    return binary_accuracy, main_class_accuracy, combined_accuracy, all_binary_labels, all_main_class_labels, all_binary_predictions, all_main_class_predictions

def transform_labels(labels, label_encoder, device):
    """Transform labels to binary and multi-class formats for the multiheaded classification model."""
    encoded_labels = label_encoder.transform(labels)
    other_cat_num = label_encoder.transform(['Other'])[0]

    # Create binary labels tensor. 1 for main classes, 0 for 'Other'
    binary_labels = torch.tensor(encoded_labels != other_cat_num, dtype=torch.long).to(device)

    # Create main class labels tensor
    main_class_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    # Replace 'other_cat_num' with -1 in 'main_class_labels'
    main_class_labels = torch.where(main_class_labels == other_cat_num, torch.tensor(-1, device=device), main_class_labels)

    # Decrement labels greater than other_cat_num by 1
    main_class_labels = torch.where(main_class_labels > other_cat_num, main_class_labels - 1, main_class_labels)
    return binary_labels, main_class_labels

def calculate_multihead_accuracy(binary_labels, main_class_labels, binary_predictions, main_class_predictions):
    '''Function that inputs the binary and main class labels and predictions of a two-headed classifier model.
    
    Process for calculating accuracy:
    -Count up true negatives.  Create a mask for true positives.
    -Pass true positive mask to main_class_predictions and count number of true positives of true positive. 
    -Add that number to true negatives and divide by number of samples in experiment.
    '''
   # Break into "Others" and "Main Class" labels
    others_mask = (torch.tensor(binary_labels) == 0)
    main_class_mask = (torch.tensor(binary_labels) == 1)

    # Count the number of true negatives that are correctly predicted
    true_neg = (torch.tensor(binary_predictions)[others_mask] == torch.tensor(binary_labels)[others_mask]).sum().item()

    # Create a mask of true positives in the binary classification head
    true_pos_mask = torch.tensor(binary_predictions)[main_class_mask] == torch.tensor(binary_labels)[main_class_mask]

    # Apply true pos mask to the multiclass predictions (only calculate accuracy on muliclass if
    # the binary head corrected predicted "Main Class")
    true_true_pos = (torch.tensor(main_class_predictions)[true_pos_mask] == torch.tensor(main_class_labels)[true_pos_mask]).sum().item()

    # Calculate the accuracy
    accuracy = (true_neg + true_true_pos) / len(binary_labels)
    return accuracy
    

def run_training_loop(model, train_dataloader, val_dataloader, label_encoder, loss_function, optimizer, device, model_type='', num_epochs=2, 
                      validation_interval=1, plot_results=True, figsize=(8,4)):
    '''A training loop to train the full model based on the training dataloader.  After a set number of epochs will run a validation set 
    through the model to determine accuracy.  Plots summary of results at the end of training and returns the loss and accuracy information'''
    train_losses = []
    validation_losses = []
    accuracy = []

    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        epoch_loss = []

        for data in train_dataloader:

            #Run the prediction
            if model_type == 'DNA_only': prediction = predict_for_DNA_only(model, data, device)
            elif model_type == 'protein_only': prediction = predict_for_protein_only(model, data, device)
            elif model_type == 'protein_and_DNA': prediction = predict_for_protein_and_DNA(model, data, device)
            else: raise ValueError(f"Invalid model_type provided: {model_type}. Expected 'DNA_only', 'protein_only', or 'protein_and_DNA'.")

            # Collect labels
            labels = data['label']
            encoded_labels = label_encoder.transform(labels)
            labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

            # Calculate loss
            loss = loss_function(prediction, labels)
            epoch_loss.append(loss.item())  # Store the loss for this batch

            # Backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss for the epoch
        average_epoch_loss = np.mean(epoch_loss)
        train_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1}, Training loss: {average_epoch_loss}")

        # Validation step at intervals
        if (epoch + 1) % validation_interval == 0:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No gradients need to be calculated
                validation_epoch_losses = []
                epoch_accuracy = []
                for data in val_dataloader:
                     #Run the prediction
                    if model_type == 'DNA_only': prediction = predict_for_DNA_only(model, data, device)
                    elif model_type == 'protein_only': prediction = predict_for_protein_only(model, data, device)
                    elif model_type == 'protein_and_DNA': prediction = predict_for_protein_and_DNA(model, data, device)
                    else: raise ValueError(f"Invalid model_type provided: {model_type}. Expected 'DNA_only', 'protein_only', or 'protein_and_DNA'.")
                    
                    labels = data['label']
                    encoded_labels = label_encoder.transform(labels)
                    labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

                    # Calculate loss
                    loss = loss_function(prediction, labels)
                    validation_epoch_losses.append(loss.item())

                    # Calculate accuracy
                    _, predicted_labels = torch.max(prediction, axis=1)
                    correct_predictions = (predicted_labels == labels).sum().item()
                    epoch_accuracy.append(correct_predictions / len(labels))

                # Average validation loss for this interval
                average_validation_loss = np.mean(validation_epoch_losses)
                validation_losses.append(average_validation_loss)
                print(f"Epoch {epoch + 1}, Validation Loss: {average_validation_loss}")

                # Average accuracy for this interval
                accuracy.append(np.mean(epoch_accuracy))
        
    if plot_results:
        plot_training_results(train_losses, validation_losses, accuracy, figsize=figsize)

    return train_losses, validation_losses, accuracy



def run_training_loop_multihead(model, train_dataloader, val_dataloader, label_encoder, optimizer, device, model_type='protein_only', 
                                num_epochs=2, validation_interval=1, plot_results=True, figsize=(8,4)):
    '''A training loop to train the full multi-headed model based on the training dataloader.  This requires calculating loss for both
    the binary classifier and multi-class classifier. After a set number of epochs will run a validation set through the model to determine 
    accuracy.  Plots summary of results at the end of training and returns the loss and accuracy information'''
    train_losses = []
    validation_losses = []
    binary_accuracy = []
    main_class_accuracy = []
    accuracy = []

    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        epoch_loss = []

        for data in train_dataloader:
            protein_embeddings = data['embeddings'].to(device)
            labels = data['label']
            binary_labels, main_class_labels = transform_labels(labels, label_encoder, device)

            # Forward pass
            binary_probs, main_class_probs = model(protein_embeddings)

            # Setup weights for different loss functions
            weight_binary = 0.5
            weight_main = 0.5

            # Calculate loss for binary classifier
            loss_binary = torch.nn.functional.binary_cross_entropy(binary_probs.squeeze(), binary_labels.float())

            # Mask for main class labels that are not 'Other'
            valid_main_class_mask = main_class_labels != -1
            if valid_main_class_mask.any(): #Check if there are any main class categories in the batch
                loss_main = torch.nn.functional.cross_entropy(main_class_probs[valid_main_class_mask], main_class_labels[valid_main_class_mask])
                loss = weight_binary*loss_binary + weight_main*loss_main
            else:
                loss = loss_binary

            epoch_loss.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_epoch_loss = np.mean(epoch_loss)
        train_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1}, Training loss: {average_epoch_loss}")

            # Validation step at intervals
        if (epoch + 1) % validation_interval == 0:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No gradients need to be calculated
                validation_epoch_losses = []
                all_binary_labels = []
                all_main_class_labels = []
                all_binary_predictions = []
                all_main_class_predictions = []

                for data in val_dataloader:
                    protein_embeddings = data['embeddings'].to(device)
                    labels = data['label']
                    binary_labels, main_class_labels = transform_labels(labels, label_encoder, device)
                    
                    #Run the prediction
                    binary_probs, main_class_probs = model(protein_embeddings)

                    # Calculate loss for binary classifier
                    loss_binary = torch.nn.functional.binary_cross_entropy(binary_probs.squeeze(), binary_labels.float())

                    # Mask for main class labels that are not 'Other'
                    valid_main_class_mask = main_class_labels != -1
                    if valid_main_class_mask.any(): #Check if there are any main class categories in the batch
                        loss_main = torch.nn.functional.cross_entropy(main_class_probs[valid_main_class_mask], main_class_labels[valid_main_class_mask])
                        loss = weight_binary*loss_binary + weight_main*loss_main
                    else:
                        loss = loss_binary
                    # Capture epoch loss
                    validation_epoch_losses.append(loss.item())

                    # Calculate accuracy
                    # Calculate correct predictions
                    predicted_binary_labels = torch.where(binary_probs > 0.5, torch.ones_like(binary_probs), torch.zeros_like(binary_probs)).squeeze()
                    _, predicted_main_class_labels = torch.max(main_class_probs[valid_main_class_mask], axis=1)

                    # Collect predictions and labels
                    all_binary_labels.extend(binary_labels.tolist())
                    all_main_class_labels.extend(main_class_labels[valid_main_class_mask].tolist())
                    all_binary_predictions.extend(predicted_binary_labels.tolist())
                    all_main_class_predictions.extend(predicted_main_class_labels.tolist())


                # Average validation loss for this interval
                average_validation_loss = np.mean(validation_epoch_losses)
                validation_losses.append(average_validation_loss)
                print(f"Epoch {epoch + 1}, Validation Loss: {average_validation_loss}")

                 # Calculate overall accuracy and add to list
                combined_accuracy = calculate_multihead_accuracy(all_binary_labels, all_main_class_labels, all_binary_predictions, all_main_class_predictions)
                accuracy.append(combined_accuracy)

    if plot_results:
        #accuracy = (binary_accuracy, main_class_accuracy)
        plot_training_results(train_losses, validation_losses, accuracy, figsize=figsize, single_or_multi_head='')

    return train_losses, validation_losses, accuracy

def plot_training_results(train_losses, validation_losses, accuracy, figsize, single_or_multi_head='single'):
    epochs = np.arange(1, len(train_losses) + 1)

    # Set up figure for losses and figure for accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot training and validation loss
    ax1.plot(epochs, train_losses, label='Training Loss', color='royalblue')
    ax1.plot(epochs, validation_losses, label='Validation Loss', color='tomato')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot training and validation accuracy
    if single_or_multi_head == 'multi':
        binary_accuracy = accuracy[0]
        main_class_accuracy = accuracy[1]
        ax2.plot(epochs, binary_accuracy, label='Binary Accuracy', color='seagreen')
        ax2.plot(epochs, main_class_accuracy, label='Main Class Accuracy', color='cyan')
    else:
        ax2.plot(epochs, accuracy, label='Training Accuracy', color='seagreen')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
def load_training_and_val_data(file_path, keep_other=False, num_classes = '_5', model_size = '_650M', num_samples = '_all', remove_DNA = '_True', curated=''):
    '''Process to load the training and validation data from a given file path string.  Loads the refseq and labels data and adjusts the labels as 
    needed for training and validation.'''
    # Folder location for each datatype
    protein_file_path = '/data/Protein_CLS_Embeddings/protein_cls_embeddings'
    DNA_file_path = '/data/DNABERT_CLS_Embeddings/DNA_cls_embeddings'
    refseq_and_labels_file_path = '/data/RefSeq_ids/refseq_ids_and_labels'

    # Add on additional classifiers based on data processing
    # Training Data File Paths
    protein_file_path_train =  file_path + protein_file_path + '_train' + num_samples + model_size + remove_DNA + curated + '.h5'
    DNA_file_path_train = file_path + DNA_file_path + '_train' + num_samples + remove_DNA + curated +'.h5'
    refseq_and_labels_file_path_train = file_path + refseq_and_labels_file_path + '_train' + num_samples + num_classes + remove_DNA + curated +'.csv'

    # Validation Data File Paths
    protein_file_path_val =  file_path + protein_file_path + '_val' + num_samples + model_size + remove_DNA + curated +'.h5'
    DNA_file_path_val = file_path + DNA_file_path + '_val' + num_samples + remove_DNA + curated +'.h5'
    refseq_and_labels_file_path_val = file_path + refseq_and_labels_file_path + '_val' + num_samples + num_classes + remove_DNA + curated +'.csv'

    # Load Training Data
    refseq_and_labels_train_df = pd.read_csv(refseq_and_labels_file_path_train)
    if not keep_other:
        refseq_and_labels_train_df = refseq_and_labels_train_df[refseq_and_labels_train_df['adjusted_host_name'] != 'Other']
    refseq_and_labels_train_df.loc[refseq_and_labels_train_df['adjusted_host_name'] == 'Mycolicibacterium smegmatis', 'adjusted_host_name'] = 'Mycolicibacterium smegmatis MC2 155'
    refseq_and_labels_train_df.loc[refseq_and_labels_train_df['adjusted_host_name'] == 'Escherichia coli K-12', 'adjusted_host_name'] = 'Escherichia coli'
    refseq_and_labels_train_df.loc[refseq_and_labels_train_df['adjusted_host_name'] == 'Salmonella enterica subsp. enterica serovar Typhimurium', 'adjusted_host_name'] = 'Salmonella enterica'
    refseq_and_labels_train_df.loc[refseq_and_labels_train_df['adjusted_host_name'] == 'Salmonella enterica subsp. enterica serovar Enteritidis', 'adjusted_host_name'] = 'Salmonella enterica'
    refseq_and_labels_train_df.loc[refseq_and_labels_train_df['adjusted_host_name'] == 'Escherichia', 'adjusted_host_name'] = 'Escherichia coli'
    refseq_ids_train = list(refseq_and_labels_train_df['refseq_id'])
    full_label_set_train = list(refseq_and_labels_train_df['adjusted_host_name'])

    # Validation Data
    refseq_and_labels_val_df = pd.read_csv(refseq_and_labels_file_path_val)
    if not keep_other:
        refseq_and_labels_val_df = refseq_and_labels_val_df[refseq_and_labels_val_df['adjusted_host_name'] != 'Other']
    refseq_and_labels_val_df.loc[refseq_and_labels_val_df['adjusted_host_name'] == 'Mycolicibacterium smegmatis', 'adjusted_host_name'] = 'Mycolicibacterium smegmatis MC2 155'
    refseq_and_labels_val_df.loc[refseq_and_labels_val_df['adjusted_host_name'] == 'Escherichia coli K-12', 'adjusted_host_name'] = 'Escherichia coli'
    refseq_and_labels_val_df.loc[refseq_and_labels_val_df['adjusted_host_name'] == 'Salmonella enterica subsp. enterica serovar Typhimurium', 'adjusted_host_name'] = 'Salmonella enterica'
    refseq_and_labels_val_df.loc[refseq_and_labels_val_df['adjusted_host_name'] == 'Salmonella enterica subsp. enterica serovar Enteritidis', 'adjusted_host_name'] = 'Salmonella enterica'
    refseq_and_labels_val_df.loc[refseq_and_labels_val_df['adjusted_host_name'] == 'Escherichia', 'adjusted_host_name'] = 'Escherichia coli'
    refseq_ids_val = list(refseq_and_labels_val_df['refseq_id'])
    full_label_set_val = list(refseq_and_labels_val_df['adjusted_host_name'])

    return protein_file_path_train, DNA_file_path_train, refseq_and_labels_file_path_train, protein_file_path_val, DNA_file_path_val, refseq_and_labels_file_path_val, refseq_ids_train, full_label_set_train, refseq_ids_val, full_label_set_val


class TrainingClass():
    """
    Base class for setting up and managing the training process for different types of data.
    
    Attributes:
        file_path (str): Base directory for data and model files.
        device (str): Device to which tensors will be sent ('cpu' or 'cuda').
        num_classes (int): Number of classes for classification.
        keep_other (bool): Flag to include or exclude the 'other' class in classification.
        model_size (str): Identifier for the size of the model being used.
    """
    def __init__(self, file_path, device, num_classes, keep_other=False, model_size='_650M', curated=''):
        """
        Initializes the TrainingClass with paths and configurations for data and models.
        """
        # Save device info
        self.device = device
        
        # Collect file names, refseq id's, and labels
        self.file_path = file_path
        num_classes_str = '_' + str(num_classes)
        self.protein_file_path_train, self.DNA_file_path_train, self.refseq_and_labels_file_path_train, \
        self.protein_file_path_val, self.DNA_file_path_val, self.refseq_and_labels_file_path_val, \
        self.refseq_ids_train, self.full_label_set_train, self.refseq_ids_val, self.full_label_set_val = \
            load_training_and_val_data(file_path, keep_other=keep_other, num_classes=num_classes_str, 
                                       model_size = model_size, num_samples = '_all', remove_DNA = '_True',
                                       curated=curated)

        #Load the encoder
        if curated == '':
            self.encoder_file_path = file_path + '/data/filtered_data/dna_protein_compiled_long_DNA_removed_50ovlp_2024_07_16.csv'
        elif curated == '_curated':
            self.encoder_file_path = file_path + '/data/filtered_data/dna_protein_compiled_curated_long_DNA_removed_50ovlp_2024_07_20.csv'
        self.label_encoder = create_label_encoder(self.encoder_file_path, num_top_hosts=num_classes, keep_other=keep_other)

    def get_data_length(self):
        print(f'Training data is length: {len(self.dataset_train)}')
        print(f'Validation data is length: {len(self.dataset_val)}')

    def run_inference(self, upsampled=True, verbose=False):
        """
        Runs the inference on validation data and returns accuracy and predictions.
        
        Parameters:
            upsampled (bool): Whether to use upsampled dataloaders for inference.
            verbose (bool): Flag to turn on detailed logging of inference process.
        
        Returns:
            tuple: Tuple containing accuracy, all labels, and all predictions.
        """
        if upsampled: 
            temp_dataloader = self.dataloader_upsampled_val
        else: 
            temp_dataloader = self.dataloader_val
        
        # Run the inference and collect results
        self.accuracy, self.all_labels, self.all_predictions = run_inference_on_data(self.model, dataloader=temp_dataloader, 
                                                                          label_encoder=self.label_encoder, device=self.device,
                                                                          model_type=self.model_type, verbose=verbose)

    def run_training(self, loss_function, optimizer, num_epochs=20, validation_interval=1, plot_results=True, figsize=(8,4)):
        """
        Conducts the training loop over a specified number of epochs.
        
        Parameters:
            loss_function: Loss function to use for training.
            optimizer: Optimizer to use for parameter updates.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs to train.
            validation_interval (int): Frequency of validation within training epochs.
            plot_results (bool): Whether to plot training results.
        
        Returns:
            tuple: Tuple containing training losses, validation losses, and validation accuracies.
        """
        
        # Run training for set number of epochs
        train_losses, val_losses, val_accuracies = run_training_loop(self.model, train_dataloader=self.dataloader_upsampled_train, val_dataloader=self.dataloader_val,
                                                                     label_encoder=self.label_encoder, loss_function=loss_function, optimizer=optimizer, device=self.device,
                                                                     model_type=self.model_type, num_epochs=num_epochs, validation_interval=validation_interval,
                                                                     plot_results=plot_results, figsize=figsize)
        return train_losses, val_losses, val_accuracies


    def confusion_matrix(self, figsize=(10,7)):
        '''Prints out a confusion matrix based on the most recently run data for inference'''
        labels_string = self.label_encoder.inverse_transform(self.all_labels)
        predictions_string = self.label_encoder.inverse_transform(self.all_predictions)
        cm = confusion_matrix(labels_string, predictions_string, labels=self.label_encoder.classes_)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_, annot_kws={"size": 14})
        plt.xlabel('Predicted Labels', fontsize=17)
        plt.ylabel('True Labels', fontsize=17)
        plt.title('Confusion Matrix', fontsize=17)
        plt.xticks(rotation=90, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_pca_analysis(self, figsize=(10,7)):
        '''Input a model trainer (which contains a dataset) and plot the PCA dimensionality
        reduction analysis on the training data'''

        # Collect embeddings and labels from the dataset
        embeddings = []
        labels = []

        # Create a DataLoader for batch processing
        dataloader = DataLoader(self.dataset_train, batch_size=64, shuffle=False)

        # Iterate over batches and extract embeddings and labels
        for batch in dataloader:
            batch_embeddings = batch['embeddings'].numpy()
            batch_labels = batch['label']

            # Append to lists
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels)

        # Convert list to NumPy arrays
        embeddings_np = np.concatenate(embeddings, axis=0)

        # Create a label encoder to convert labels to integers
        label_encoder = LabelEncoder()
        labels_np = label_encoder.fit_transform(labels)

        # Run PCA Analysis
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_np)

        # Plot results
        plt.figure(figsize=figsize)
        scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels_np, cmap='viridis', s=5)
        plt.title('PCA of Embeddings')
        plt.xlabel('Principal Component 1', fontsize=13)
        plt.ylabel('Principal Component 2', fontsize=13)

        # Getting unique labels and their corresponding colors
        unique_labels = list(np.unique(labels_np))
        colors = [scatter.cmap(s) for s in scatter.norm(unique_labels)]

        # Creating the legend
        legend_labels = list(label_encoder.inverse_transform(unique_labels))
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
        plt.legend(handles=legend_handles, labels=legend_labels, title="Categories")
        plt.show()

class ProteinOnlyTrainingClass(TrainingClass):
    """
    Training class specialized for handling protein-only datasets and model.
    Inherits from TrainingClass and adds specific configurations for protein data handling.
    """
    def __init__(self, file_path, device, num_classes, dropout=0.5, protein_combo_style='mean', hidden_layer_sizes=[512,256,128,64], ESM_model_size='facebook/esm2_t33_650M_UR50D', 
                 keep_other=False, model_size='_650M', init_mode='', batch_size=16, balance_other=False, curated=''):
         # Call the constructor of TrainingClass
        super().__init__(file_path, device, num_classes=num_classes, keep_other=keep_other, model_size=model_size, curated=curated)

        # Save model type
        self.model_type = 'protein_only'

        # Create datasets and dataloaders
        if protein_combo_style == 'mean':
            # Training Data
            self.dataset_train = ProteinAverageEmbeddingsDataset(self.protein_file_path_train, self.refseq_ids_train, self.full_label_set_train)
            self.dataloader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=False)
            #Validation Data
            self.dataset_val = ProteinAverageEmbeddingsDataset(self.protein_file_path_val, self.refseq_ids_val, self.full_label_set_val)
            self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)

        elif protein_combo_style == 'max':
            # Training Data
            self.dataset_train = ProteinMaxEmbeddingsDataset(self.protein_file_path_train, self.refseq_ids_train, self.full_label_set_train)
            self.dataloader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=False)
            #Validation Data
            self.dataset_val = ProteinMaxEmbeddingsDataset(self.protein_file_path_val, self.refseq_ids_val, self.full_label_set_val)
            self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)

        # Create upsampled dataloader
        # Training Data
        self.dataloader_upsampled_train = create_weighted_sampler_dataloader(self.dataset_train, self.label_encoder, self.full_label_set_train, batch_size=batch_size, shuffle=False, balance_other=balance_other)
        # Validation Data
        self.dataloader_upsampled_val = create_weighted_sampler_dataloader(self.dataset_val, self.label_encoder, self.full_label_set_val, batch_size=batch_size, shuffle=False, balance_other=balance_other)


        # Load the protein model
        self.ESM_model = AutoModel.from_pretrained(ESM_model_size).to(device)
        self.model = ProteinOnlyClassifier(self.ESM_model, num_classes=len(self.label_encoder.classes_), 
                                           hidden_layer_sizes=hidden_layer_sizes, dropout=dropout, init_mode=init_mode).to(device)


class DNAOnlyTrainingClass(TrainingClass):
    """
    Training class specialized for handling DNA-only datasets and model.
    Inherits from TrainingClass and adds specific configurations for DNA data handling.
    """
    def __init__(self, file_path, location, device, num_classes, dropout=0.5, hidden_layer_sizes=[512,256,128,64], keep_other=False, model_size='_650M',
                 batch_size=16, curated=''):
        # Call the constructor of TrainingClass
        super().__init__(file_path, device, num_classes=num_classes, keep_other=keep_other, model_size=model_size, curated=curated)

        # Save model type
        self.model_type = 'DNA_only'
        
        # Create datasets and dataloaders
        # Training Data
        self.dataset_train = CLSEmbeddingsDataset(self.DNA_file_path_train, self.refseq_ids_train, self.full_label_set_train)
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=False)
        # Validation Data
        self.dataset_val = CLSEmbeddingsDataset(self.DNA_file_path_val, self.refseq_ids_val, self.full_label_set_val)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)

        # Create upsampled dataloader
        # Training Data
        self.dataloader_upsampled_train = create_weighted_sampler_dataloader(self.dataset_train, self.label_encoder, self.full_label_set_train, batch_size=batch_size, shuffle=False)
        # Validation Data
        self.dataloader_upsampled_val = create_weighted_sampler_dataloader(self.dataset_val, self.label_encoder, self.full_label_set_val, batch_size=batch_size, shuffle=False)

        # Create new instantiations of the DNA model
        self.DNA_model, _ = load_DNA_model(location=location, device=device)
        self.model = DNAOnlyClassifier(self.DNA_model, num_classes=len(self.label_encoder.classes_), 
                                            hidden_layer_sizes=hidden_layer_sizes, dropout=dropout).to(device)
        
class ProteinAndDNATrainingClass(TrainingClass):
    """
    Training class for handling both DNA and protein datasets - builds a combined model.
    Inherits from TrainingClass and adds specific configurations for the combined protein and
    DNA data handling
    """
    def __init__(self, file_path, location, device, num_classes, protein_combo_style='mean', hidden_layer_sizes=[512,256,128,64], 
                 keep_other=False, model_size='_650M', ESM_model_size='facebook/esm2_t33_650M_UR50D', dropout=0.5, init_mode='',
                 batch_size=16, curated=''):
        # Call the constructor of TrainingClass
        super().__init__(file_path, device, num_classes=num_classes, keep_other=keep_other, model_size=model_size, curated=curated)

        # Save model type
        self.model_type = 'protein_and_DNA'
        
        # Create datasets and dataloaders
        # Protein Data
        if protein_combo_style == 'mean':
            # Training Data
            self.protein_dataset_train = ProteinAverageEmbeddingsDataset(self.protein_file_path_train, self.refseq_ids_train, self.full_label_set_train)
            self.protein_dataloader_train = DataLoader(self.protein_dataset_train, batch_size=batch_size, shuffle=False)
            #Validation Data
            self.protein_dataset_val = ProteinAverageEmbeddingsDataset(self.protein_file_path_val, self.refseq_ids_val, self.full_label_set_val)
            self.protein_dataloader_val = DataLoader(self.protein_dataset_val, batch_size=batch_size, shuffle=False)

        elif protein_combo_style == 'max':
            # Training Data
            self.protein_dataset_train = ProteinMaxEmbeddingsDataset(self.protein_file_path_train, self.refseq_ids_train, self.full_label_set_train)
            self.protein_dataloader_train = DataLoader(self.protein_dataset_train, batch_size=batch_size, shuffle=False)
            #Validation Data
            self.protein_dataset_val = ProteinMaxEmbeddingsDataset(self.protein_file_path_val, self.refseq_ids_val, self.full_label_set_val)
            self.protein_dataloader_val = DataLoader(self.protein_dataset_val, batch_size=batch_size, shuffle=False)

        #DNA Data
        # Training Data
        self.DNA_dataset_train = CLSEmbeddingsDataset(self.DNA_file_path_train, self.refseq_ids_train, self.full_label_set_train)
        self.DNA_dataloader_train = DataLoader(self.DNA_dataset_train, batch_size=batch_size, shuffle=False)
        # Validation Data
        self.DNA_dataset_val = CLSEmbeddingsDataset(self.DNA_file_path_val, self.refseq_ids_val, self.full_label_set_val)
        self.DNA_dataloader_val = DataLoader(self.DNA_dataset_val, batch_size=batch_size, shuffle=False)

        # Combine the DNA and Protein Datasets
        # Training Data
        self.dataset_train = CombinedDataset(self.DNA_dataset_train, self.protein_dataset_train)
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=False)
        # Validation Data
        self.dataset_val = CombinedDataset(self.DNA_dataset_val, self.protein_dataset_val)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)

        # Create upsampled dataloader
        # Training Data
        self.dataloader_upsampled_train = create_weighted_sampler_dataloader(self.dataset_train, self.label_encoder, self.full_label_set_train, 
                                                                             batch_size=batch_size, shuffle=False)
        # Validation Data
        self.dataloader_upsampled_val = create_weighted_sampler_dataloader(self.dataset_val, self.label_encoder, self.full_label_set_val, 
                                                                           batch_size=batch_size, shuffle=False)

        # Create new instantiations of the DNA model
        self.DNA_model, _ = load_DNA_model(location=location, device=device)
        self.ESM_model = AutoModel.from_pretrained(ESM_model_size).to(device)
        self.model = CustomClassifier(self.DNA_model, self.ESM_model, num_classes=len(self.label_encoder.classes_), 
                                      hidden_layer_sizes=hidden_layer_sizes, cls_token_id=1, dropout=dropout, init_mode=init_mode).to(device)
        
class DualHeadProteinOnlyTrainingClass(ProteinOnlyTrainingClass):
    """
    Training class specialized for handling protein-only datasets and model.
    Inherits from TrainingClass and adds specific configurations for protein data handling.
    Changes output architeture to include two heads - one for binary classification of "other" class, one for classification
    among remaining "main classes".
    """
    def __init__(self, file_path, device, num_classes, dropout=0.5, protein_combo_style='mean', hidden_layer_sizes=[512,256,128,64], ESM_model_size='facebook/esm2_t33_650M_UR50D', 
                 keep_other=True, model_size='_650M', init_mode='', batch_size=16, balance_other=True, curated=''):
         # Call the constructor of TrainingClass
        super().__init__(file_path, device, num_classes=num_classes, dropout=dropout, protein_combo_style=protein_combo_style, hidden_layer_sizes=hidden_layer_sizes,
                         ESM_model_size=ESM_model_size, keep_other=keep_other, model_size=model_size, init_mode=init_mode, batch_size=batch_size,
                         balance_other=balance_other, curated=curated)

        # Update model to dual headed architecture
        self.model = DualHeadProteinOnlyClassifier(self.ESM_model, num_main_classes=(len(self.label_encoder.classes_)-1), 
                                           hidden_layer_sizes=hidden_layer_sizes, dropout=dropout, init_mode=init_mode).to(device)
        self.model_type = 'protein_only'
        
        self.label_encoder_cm = create_label_encoder(self.encoder_file_path, num_top_hosts=num_classes, keep_other=False)
    
    def run_inference(self, upsampled=True, verbose=False):
        """
        Runs the inference on validation data and returns accuracy and predictions.
        Updated from parent class to work with dualheaded architecture
        """
        if upsampled: 
            temp_dataloader = self.dataloader_upsampled_val
        else: 
            temp_dataloader = self.dataloader_val
        
        # Run the inference and collect results
        self.binary_accuracy, self.main_class_accuracy, self.combined_accuracy, self.all_binary_labels, \
        self.all_main_class_labels, self.all_binary_predictions, self.all_main_class_predictions\
             = run_inference_on_data_multihead(self.model, dataloader=temp_dataloader, label_encoder=self.label_encoder, 
                                                device=self.device, model_type=self.model_type, verbose=verbose)
        
        #return self.binary_accuracy, self.main_class_accuracy, self.combined_accuracy, self.all_binary_labels, self.all_main_class_labels, self.all_binary_predictions, self.all_main_class_predictions

    def run_training(self, optimizer, num_epochs=20, validation_interval=1, plot_results=True, figsize=(8,4), loss_function=''):
        """
        Conducts the training loop over a specified number of epochs.
        Updated from parent class to work with dualheaded architecture

        """
        
        # Run training for set number of epochs
        train_losses, val_losses, accuracy = run_training_loop_multihead(self.model, train_dataloader=self.dataloader_upsampled_train, val_dataloader=self.dataloader_val,
                                                                     label_encoder=self.label_encoder, optimizer=optimizer, device=self.device,
                                                                     model_type=self.model_type, num_epochs=num_epochs, validation_interval=validation_interval,
                                                                     plot_results=plot_results, figsize=figsize)
        return train_losses, val_losses, accuracy
       

    def confusion_matrix(self, figsize=(10,7)):
        '''Prints out a confusion matrix based on the most recently run data for inference
        Updated from paret class to work with dualehaded architecture'''
        
        # First confusion matrix
        labels_string = self.label_encoder_cm.inverse_transform(self.all_main_class_labels)
        predictions_string = self.label_encoder_cm.inverse_transform(self.all_main_class_predictions)
        cm1 = confusion_matrix(labels_string, predictions_string, labels=self.label_encoder_cm.classes_)

        # Second confusion matrix
        cm2 = confusion_matrix(self.all_binary_labels, self.all_binary_predictions, labels=[0,1])

        # Setup figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Heatmap for the first confusion matrix
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder_cm.classes_, yticklabels=self.label_encoder_cm.classes_, ax=ax1, annot_kws={"size": 14})
        ax1.set_xlabel('Predicted Labels', fontsize=17)
        ax1.set_ylabel('True Labels', fontsize=17)
        ax1.set_title('Main Class Confusion Matrix', fontsize=17)
        ax1.tick_params(axis='x', rotation=90, labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)

        # Heatmap for the second confusion matrix
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Other', 'Main Class'], yticklabels=['Other', 'Main Class'], ax=ax2, annot_kws={"size": 14})
        ax2.set_xlabel('Predicted Labels', fontsize=17)
        ax2.set_ylabel('True Labels', fontsize=17)
        ax2.set_title('Binary Class Confusion Matrix', fontsize=17)
        ax2.tick_params(axis='x', rotation=90, labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        plt.show()

