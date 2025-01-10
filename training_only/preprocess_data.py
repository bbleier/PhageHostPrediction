import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import math
import gc

def clean_and_filter_df(df):
    '''Function to clean the Virus-Host DB Data and return a 
    dataframe with a present cutoff of 20 examples/bacteria'''
    #Add a column that pulls out the host type
    def parse_host_type(x):
        x = str(x)
        return x.split(';')[0]
    
    def select_first_refseqid(x):
        try: 
            first_name = x.split(',')[0]
        except:
            first_name = x
        return first_name

    df['host_type'] = df['host lineage'].map(parse_host_type)

    #Trim the dataframe to bacteria only
    df_bacteria = df[df['host_type'] == 'Bacteria']

    #Set up a threshold for minimum number of interactions per bacteria
    cutoff_value = 20
    df_groupby_bacteria = df_bacteria.groupby('host name')
    sorted_bacteria_count = np.sort(df_groupby_bacteria['host name'].count().values)[::-1]
    count = df_groupby_bacteria['host name'].count()
    filtered_names = list(count[count >= cutoff_value].index)

    df_bacteria_filtered = df_bacteria[df_bacteria['host name'].isin(filtered_names)]

    #Select the first refseq id in the rare case that multiple id's are listed in this column
    df_bacteria_filtered.loc[:, 'refseq id'] = df_bacteria_filtered['refseq id'].map(select_first_refseqid)

    #Summarize
    print(f'The filtered dataset includes {df_bacteria_filtered.shape[0]} samples from {len(filtered_names)} different bacteria')
    print(f'All bacteria have at least {cutoff_value} samples each')

    return df_bacteria_filtered


def load_file(file_path):
    '''Loads the dataframe of protein and DNA information from a saved csv file'''
    # Load dataframe from the saved csv file
    grouped_data_df = pd.read_csv(file_path)
    
    # Convert JSON strings back to lists
    grouped_data_df['protein_sequence'] = grouped_data_df['protein_sequence'].apply(json.loads)
    grouped_data_df['protein_label'] = grouped_data_df['protein_label'].apply(json.loads)
    #grouped_data_df['protein_sequence'] = grouped_data_df['protein_sequence'].apply(json.loads) #Run twice b/c of incorrect storage?

    return grouped_data_df

def calculate_chunks(x, tokenizer, overlap_size = 50):
    '''Calculate the number of 510 unit chunks (510 tokens + CLS + EOS) that
    a given DNA strand will be broken into.  Goal is to find strands with >100
    chunks so that they can be removed from the dataset'''
    num_tokens = len(tokenizer.tokenize(x))
    num_chunks = (num_tokens - 510)/(510 - overlap_size) + 1 # [(T-C)/(C-O) + 1] Token, Chunk, Overlap
    return math.ceil(num_chunks)

def remove_long_DNA(df, tokenizer, overlap_size=50):
    '''Removes samples from dataset if DNA is too long'''
    #add column that calculates number of chunks
    df['number_of_chunks'] = df['gene_sequence'].apply(lambda x: calculate_chunks(x, tokenizer, overlap_size=50))
    # Remove DNA that is too long
    df = df[df['number_of_chunks'] <= 100]
    return df

def load_data(file_path, remove_long_DNA_flag=False, tokenizer=None, overlap_size=50):
    '''Loads a dataframe of preprocessed data containing refseq_id, protein labels,
    protein sequences, DNA sequences, and output label
    
    Option to remove samples from dataset if the DNA is too long (>100 chunks of 
    tokenized data)'''
    df = load_file(file_path)

    if remove_long_DNA_flag:
        df = remove_long_DNA(df, tokenizer, overlap_size=overlap_size)
    
    return df

def load_data_in_batches(file_path, batch_size = 500, remove_long_DNA_flag=False, tokenizer=None, overlap_size=50):
    '''Function to batch the loading of data due to overwhelm in RAM, specifically for removing long DNA (DNA longer
    than 100 chunks worth of embeddings).  Will execute based on predefined batch_size'''
    # load the full dataframe
    df = load_file(file_path)

    # Determine the number of batches required to process all the data
    num_samples = len(df)
    num_batches = math.ceil(num_samples/batch_size)
    all_batches = []

    # Identify and remove data with DNA that is too long to fit into DNABERT model (>100 chunks of 510 tokens)
    if remove_long_DNA_flag:
        for i in range(num_batches):
            df_batch = df[i*batch_size:(i+1)*batch_size].copy()
            df_batch = remove_long_DNA(df_batch, tokenizer, overlap_size=overlap_size)
            all_batches.append(df_batch)
            gc.collect() # Explicity call garbage collector to ensure memory efficiency
            print(f'Completed batch {i+1} out of {num_batches} batches')
        df_compiled = pd.concat(all_batches, axis=0)

        return df_compiled
    else: return df


def split_data(df, train_frac = 0.8, val_frac = 0.1, random_state=1234):
    '''Split the original data into train, validate, and test data.  Set to an 80/10/10 split,
    unless otherwise changed.  Shuffles the data in the process.'''
    #Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state)

    # Set the ratios for splitting into train/test/val sets
    train_size = int(len(df) * 0.8)
    validation_size = int(len(df) * 0.1)
    test_size = len(df) - train_size - validation_size

    # Split the data into training and temporary set (temporary set will be split into validation and test)
    train_df, temp_df = train_test_split(df_shuffled, train_size=train_size, random_state=1234)

    # Split the temporary set into validation and test sets
    val_df, test_df = train_test_split(temp_df, test_size=test_size, random_state=1234)

    #Capture the refseq labels to re-align data order later during processing
    train_refseq_labels = train_df['refseq_id'].tolist()
    val_refseq_labels = val_df['refseq_id'].tolist()
    test_refseq_labels = test_df['refseq_id'].tolist()

    return train_df, val_df, test_df, train_refseq_labels, val_refseq_labels, test_refseq_labels

def trim_to_top_n_hosts(df, num_top_hosts):
    '''Take the dataframe with all samples and identify the top n hosts by 
    count.  Keep the labeled data for samples within the top n hosts, but otherwise
    change the label name to "Other"'''

    # Identify the top hosts
    ranked_hosts_df = df.groupby('host_name')['refseq_id'].count().sort_values(ascending=False).reset_index()
    top_n_hosts = list(ranked_hosts_df[0:num_top_hosts]['host_name'])

    def rename_hosts_based_on_top_n_hosts(x, top_n_hosts_list):
        '''Perform the action of renaming the label to "Other" if it is not among the 
        top n hosts'''
        if x in top_n_hosts_list:
            x = x
        else: x = 'Other'
        return x

    # Add new column with adjusted host name based on top n hosts
    df['adjusted_host_name'] = df['host_name'].apply(lambda x: rename_hosts_based_on_top_n_hosts(x, top_n_hosts))

    return df

def create_label_encoder(file_path, num_top_hosts=19, keep_other=True):
    '''Determines the list of possible labels based on the number of top hosts
    that are selected to be in the dataset.  Then creates a label_encoder based
    on the selected set and returns the encoder for future use'''

    # Pull dataframe and add top hosts.  Then collect the list of top hosts.
    df = load_data(file_path)
    df = trim_to_top_n_hosts(df, num_top_hosts = num_top_hosts)
    labels_list = list(set(list(df['adjusted_host_name'])))

    if not keep_other:
        labels_list.remove('Other')
    
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit label encoder and return encoded labels
    encoded_labels = label_encoder.fit_transform(labels_list)

    return label_encoder

