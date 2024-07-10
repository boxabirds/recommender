import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
import time
import pickle

# Step 0: Download and extract the MovieLens 20M Dataset
def download_and_extract_dataset(url, save_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    if not os.path.exists(save_path):
        print(f"Downloading MovieLens 20M Dataset from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=save_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                f.write(data)
    
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"Dataset extracted to {extract_path}")

# Step 1: Load the MovieLens 20M Dataset
def load_data(ratings_file, movies_file, tags_file):
    # Read CSV in chunks to handle large files
    ratings_chunks = pd.read_csv(ratings_file, chunksize=1000000)
    ratings = pd.concat(ratings_chunks)
    
    movies = pd.read_csv(movies_file)
    tags_chunks = pd.read_csv(tags_file, chunksize=1000000)
    tags = pd.concat(tags_chunks)
    
    return ratings, movies, tags

# Step 2: Preprocess the data
def preprocess_data(ratings, movies, tags):
    # Merge ratings with movies to get movie titles
    df = pd.merge(ratings, movies[['movieId', 'title']], on='movieId', how='left')
    
    # Merge with tags
    df = pd.merge(df, tags[['userId', 'movieId', 'tag']], on=['userId', 'movieId'], how='left')
    
    # Handle missing values
    df['tag'] = df['tag'].fillna('Unknown')  # Changed to avoid the FutureWarning
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    return df

# Step 3: Feature extraction
def extract_features(df):
    # Encode categorical variables
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    tag_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['userId'])
    df['movie_id'] = movie_encoder.fit_transform(df['movieId'])
    df['tag_id'] = tag_encoder.fit_transform(df['tag'])
    
    # Create user-item interaction sparse matrix
    user_item_matrix = csr_matrix((df['rating'], (df['user_id'], df['movie_id'])))
    
    # Create item-tag sparse matrix
    item_tag_matrix = csr_matrix((np.ones(len(df)), (df['movie_id'], df['tag_id'])))
    
    return df, user_item_matrix, item_tag_matrix, user_encoder, movie_encoder, tag_encoder

# Step 4: Split data into train and test sets
def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

# Main function to run the data preparation process
def prepare_data(dataset_url, save_path, extract_path):
    # Check if output files already exist
    output_files = ["full_data.pkl", "train_data.pkl", "test_data.pkl", "user_item_matrix.npz", "item_tag_matrix.npz"]
    if all(os.path.exists(file) for file in output_files):
        print("Output files already exist. Terminating process.")
        return
    
    # Download and extract dataset
    start_time = time.time()
    download_and_extract_dataset(dataset_url, save_path, extract_path)
    print(f"Download and extraction time: {time.time() - start_time:.2f} seconds")
    
    # Set file paths
    ratings_file = os.path.join(extract_path, "ml-20m", "ratings.csv")
    movies_file = os.path.join(extract_path, "ml-20m", "movies.csv")
    tags_file = os.path.join(extract_path, "ml-20m", "tags.csv")
    
    # Load data
    start_time = time.time()
    ratings, movies, tags = load_data(ratings_file, movies_file, tags_file)
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    
    # Preprocess data
    start_time = time.time()
    df = preprocess_data(ratings, movies, tags)
    print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")
    
    # Extract features
    start_time = time.time()
    df, user_item_matrix, item_tag_matrix, user_encoder, movie_encoder, tag_encoder = extract_features(df)
    print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")
    
    # Split data
    start_time = time.time()
    train_df, test_df = split_data(df)
    print(f"Data splitting time: {time.time() - start_time:.2f} seconds")
    
    # Save results to disk
    with open("full_data.pkl", "wb") as f:
        pickle.dump(df, f)
    with open("train_data.pkl", "wb") as f:
        pickle.dump(train_df, f)
    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_df, f)
    np.savez("user_item_matrix.npz", data=user_item_matrix.data, indices=user_item_matrix.indices, indptr=user_item_matrix.indptr, shape=user_item_matrix.shape)
    np.savez("item_tag_matrix.npz", data=item_tag_matrix.data, indices=item_tag_matrix.indices, indptr=item_tag_matrix.indptr, shape=item_tag_matrix.shape)
    
    return {
        'full_data': df,
        'train_data': train_df,
        'test_data': test_df,
        'user_item_matrix': user_item_matrix,
        'item_tag_matrix': item_tag_matrix,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'tag_encoder': tag_encoder
    }

def verify_preprocessing(df, user_item_matrix, item_tag_matrix):
    verification_results = {}

    # Check for missing values
    verification_results['missing_values'] = df.isnull().sum().to_dict()

    # Check unique users, movies, and tags
    verification_results['unique_users'] = df['userId'].nunique()
    verification_results['unique_movies'] = df['movieId'].nunique()
    verification_results['unique_tags'] = df['tag'].nunique()

    # Check rating range
    verification_results['rating_range'] = (df['rating'].min(), df['rating'].max())

    # Check user-item matrix shape
    verification_results['user_item_matrix_shape'] = user_item_matrix.shape

    # Check item-tag matrix shape
    verification_results['item_tag_matrix_shape'] = item_tag_matrix.shape

    # Check if timestamps are in correct format
    verification_results['timestamp_range'] = (df['timestamp'].min(), df['timestamp'].max())

    return verification_results

def prepare_and_verify_data(dataset_url, save_path, extract_path):
    # Check if output files already exist
    output_files = ["full_data.pkl", "train_data.pkl", "test_data.pkl", "user_item_matrix.npz", "item_tag_matrix.npz"]
    if all(os.path.exists(file) for file in output_files):
        print("Output files already exist. Terminating process.")
        return
    
    # Download and extract dataset
    start_time = time.time()
    download_and_extract_dataset(dataset_url, save_path, extract_path)
    print(f"Download and extraction time: {time.time() - start_time:.2f} seconds")
    
    # Set file paths
    ratings_file = os.path.join(extract_path, "ml-20m", "ratings.csv")
    movies_file = os.path.join(extract_path, "ml-20m", "movies.csv")
    tags_file = os.path.join(extract_path, "ml-20m", "tags.csv")
    
    # Load data
    start_time = time.time()
    ratings, movies, tags = load_data(ratings_file, movies_file, tags_file)
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")
    
    # Preprocess data
    start_time = time.time()
    df = preprocess_data(ratings, movies, tags)
    print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")
    
    # Extract features
    start_time = time.time()
    df, user_item_matrix, item_tag_matrix, user_encoder, movie_encoder, tag_encoder = extract_features(df)
    print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")
    
    # Verify preprocessing
    start_time = time.time()
    verification_results = verify_preprocessing(df, user_item_matrix, item_tag_matrix)
    print(f"Verification time: {time.time() - start_time:.2f} seconds")
    
    # Split data
    start_time = time.time()
    train_df, test_df = split_data(df)
    print(f"Data splitting time: {time.time() - start_time:.2f} seconds")
    
    # Save results to disk
    # print out where the files are saved
    with open("full_data.pkl", "wb") as f:
        pickle.dump(df, f)
    with open("train_data.pkl", "wb") as f:
        pickle.dump(train_df, f)
    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_df, f)
    np.savez("user_item_matrix.npz", data=user_item_matrix.data, indices=user_item_matrix.indices, indptr=user_item_matrix.indptr, shape=user_item_matrix.shape)
    np.savez("item_tag_matrix.npz", data=item_tag_matrix.data, indices=item_tag_matrix.indices, indptr=item_tag_matrix.indptr, shape=item_tag_matrix.shape)
    
    # print out where all the files are being saved
    print(f"Full data saved to {os.path.abspath('full_data.pkl')}")
    
    return {
        'full_data': df,
        'train_data': train_df,
        'test_data': test_df,
        'user_item_matrix': user_item_matrix,
        'item_tag_matrix': item_tag_matrix,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'tag_encoder': tag_encoder,
        'verification_results': verification_results
    }

if __name__ == "__main__":
    dataset_url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    save_path = "ml-20m.zip"
    extract_path = "."
    
    # print out what this script is doing -- its inputs and outputs
    print(f"Preparing and verifying data from {dataset_url}...")



    prepared_data = prepare_and_verify_data(dataset_url, save_path, extract_path)
    
    if prepared_data:
        print("Data preparation completed.")
        print(f"Full dataset shape: {prepared_data['full_data'].shape}")
        print(f"Training set shape: {prepared_data['train_data'].shape}")
        print(f"Test set shape: {prepared_data['test_data'].shape}")
        print(f"User-Item matrix shape: {prepared_data['user_item_matrix'].shape}")
        print(f"Item-Tag matrix shape: {prepared_data['item_tag_matrix'].shape}")
        
        print("\nVerification Results:")
        for key, value in prepared_data['verification_results'].items():
            print(f"{key}: {value}")

        # Additional checks
        print("\nAdditional Checks:")
        print(f"Number of ratings: {len(prepared_data['full_data'])}")
        print(f"Number of unique movies: {prepared_data['verification_results']['unique_movies']}")
        print(f"Number of unique users: {prepared_data['verification_results']['unique_users']}")
        print(f"Number of unique tags: {prepared_data['verification_results']['unique_tags']}")
        print(f"Rating range: {prepared_data['verification_results']['rating_range']}")
        print(f"Timestamp range: {prepared_data['verification_results']['timestamp_range']}")

        # Check for any remaining missing values
        missing_values = prepared_data['verification_results']['missing_values']
        if any(missing_values.values()):
            print("\nWarning: There are still missing values in the dataset:")
            for column, count in missing_values.items():
                if count > 0:
                    print(f"{column}: {count} missing values")
        else:
            print("\nNo missing values found in the dataset.")

        # Print sparsity of matrices
        user_item_sparsity = 1.0 - (prepared_data['user_item_matrix'].nnz / 
                                    (prepared_data['user_item_matrix'].shape[0] * prepared_data['user_item_matrix'].shape[1]))
        item_tag_sparsity = 1.0 - (prepared_data['item_tag_matrix'].nnz / 
                                   (prepared_data['item_tag_matrix'].shape[0] * prepared_data['item_tag_matrix'].shape[1]))
        print(f"\nUser-Item matrix sparsity: {user_item_sparsity:.4f}")
        print(f"Item-Tag matrix sparsity: {item_tag_sparsity:.4f}")