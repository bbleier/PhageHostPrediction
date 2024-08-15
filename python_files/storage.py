import h5py 
import os   
import torch 
import numpy as np 
import math
import sys

from .feature_extraction import process_protein_to_CLS_embeddings 



# Write the embeddings to an h5py file for temporary storage
def store_embeddings_in_hdf5_file(hdf5_path, embeddings_by_phage, attention_mask_by_phage, verbose=True):
    '''
    Stores CLS embeddings and their corresponding attention masks in an HDF5 file, compressing the data for efficiency.

    This function opens or creates an HDF5 file at the specified path and writes embeddings and attention masks
    for each phage into separate groups within the file. The embeddings and masks are stored in datasets with
    refseq_id as the reference
    '''
     # Check if the file exists
    file_mode = 'a' if os.path.exists(hdf5_path) else 'w'

    with h5py.File(hdf5_path, file_mode) as hdf:
        for refseq_id in embeddings_by_phage.keys():
            # Convert PyTorch tensors to NumPy arrays
            embeddings_array = embeddings_by_phage[refseq_id].detach().cpu().numpy()  # Ensure it's detached and on CPU
            attention_mask_array = attention_mask_by_phage[refseq_id].detach().cpu().numpy()

            # Create a group for each refseq_id
            group = hdf.create_group(refseq_id)

            # Create datasets within this group
            group.create_dataset('embeddings', data=embeddings_array, compression="gzip", compression_opts=9)
            group.create_dataset('attention_mask', data=attention_mask_array, compression="gzip", compression_opts=9)
    if verbose:
        print('file saved successfully')

def process_protein_to_CLS_and_store_hdf5_batched(hdf5_path, protein_dataloader, protein_model, device='', batch_size=100):
    '''Processes the protein information in batches of size batch_size to help with saving process. Once samples in a batch have
    been run, it will save to the hdf5 file and continue to the next batch of samples
    
    Must use dataloader batch size = 1.  Each sample has 10 proteins which will be fed to the protein BERT, which will already
    have dimensions (mxn) where m is the number of proteins and n is the number of amino acids tokens per protein'''
    
    num_batches = math.ceil(len(protein_dataloader)/batch_size)
    dataloader_iterator = iter(protein_dataloader)

    # Iterate through the batches of data that will be saved to the hdf5 file
    for i in range(num_batches):
        embeddings_by_phage = {}
        attention_mask_by_phage = {}

        # Iterate through the data within a batch.  Pulled by dataloader batches.
        # Dataloader must have a batch size = 1
        for j in range(batch_size):
            try:
                data = next(dataloader_iterator)
            except StopIteration: break # For the last batch, break to storage function if no samples remain
            
            # Extract the tokenized chunks, downstream attention mask,
            # and refseq_id from dataloader
            sequences = data['protein_sequences'].squeeze().to(device)
            attention_mask = data['attention_mask'].squeeze().to(device)
            downstream_attention_mask = data['downstream_attention_mask'].squeeze().to(device)
            refseq_id = data['refseq_id'][0]

            # Run sample through protein model
            with torch.no_grad():
                output = protein_model(sequences, attention_mask=attention_mask)
                cls_output = output['last_hidden_state'][:,0,:]

            # Capture the data
            embeddings_by_phage[refseq_id] = cls_output
            attention_mask_by_phage[refseq_id] = downstream_attention_mask

            # Reduce memory imprint
            del cls_output
            torch.cuda.empty_cache()

            if (j+1) % 10 == 0:
                print(f'Completed phage number {j+1} out of {batch_size}, in batch {i+1} out of {num_batches}')

        # Store this batch of the embeddings
        store_embeddings_in_hdf5_file(hdf5_path, embeddings_by_phage, attention_mask_by_phage, verbose=False)
        print(f'Successfully stored data from batch {i+1} out of {num_batches}')  


def process_DNA_to_CLS_and_store_hdf5_batched(hdf5_path, DNA_dataloader, DNA_model, device='', batch_size=100):
    '''Processes the DNA information in batches of size batch_size to help with saving process. Once samples in a batch have
    been run, it will save to the hdf5 file and continue to the next batch of samples
    
    Must use dataloader batch size = 1.  Each sample has 100 DNA chunks which will be fed to DNABERT, which will already
    have dimensions (mxn) where m is the number of DHA chunks and n is the number nucleotide tokens per chunk'''
    
    num_batches = math.ceil(len(DNA_dataloader)/batch_size)
    dataloader_iterator = iter(DNA_dataloader)

    # Iterate through the batches of data that will be saved to the hdf5 file
    for i in range(num_batches):
        embeddings_by_phage = {}
        attention_mask_by_phage = {}

        # Iterate through the data within a batch.  Pulled by dataloader batches.
        # Dataloader must have a batch size = 1
        for j in range(batch_size):
            try:
                data = next(dataloader_iterator)
            except StopIteration: break # For the last batch, break to storage function if no samples remain
            
            # Extract the tokenized chunks, downstream attention mask,
            # and refseq_id from dataloader

            sequences = data['gene_sequence'].squeeze().to(device)
            attention_mask = data['attention_mask'].squeeze().to(device)
            downstream_attention_mask = data['downstream_attention_mask'].squeeze().to(device)
            refseq_id = data['refseq_id'][0]

            # Process in smaller sub-batches of 10
            sub_batch_size = 10
            embeddings = []
            with torch.no_grad():
                for k in range(0, len(sequences), sub_batch_size):
                    sub_batch_sequences = sequences[k:k + sub_batch_size]
                    sub_batch_attention_mask = attention_mask[k:k + sub_batch_size]
                    output = DNA_model(sub_batch_sequences, attention_mask=sub_batch_attention_mask)  # Process the sub-batch through DNABERT
                    cls_output = output[0][:,0,:].detach()
                    embeddings.append(cls_output)
            
            # Concatenate embeddings from subbatches into the same tensor
            concatenated_embeddings = torch.cat(embeddings, dim=0)

            # Capture the data
            embeddings_by_phage[refseq_id] = concatenated_embeddings
            attention_mask_by_phage[refseq_id] = downstream_attention_mask

            # Reduce memory imprint
            del cls_output
            torch.cuda.empty_cache()

            if (j+1) % 10 == 0:
                print(f'Completed phage number {j+1} out of {batch_size}, in batch {i+1} out of {num_batches}')

        # Store this batch of the embeddings
        store_embeddings_in_hdf5_file(hdf5_path, embeddings_by_phage, attention_mask_by_phage, verbose=False)
        print(f'Successfully stored data from batch {i+1} out of {num_batches}')  
