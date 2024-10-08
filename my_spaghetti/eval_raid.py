from raid import run_detection, run_evaluation
from raid.utils import load_data
from DualDetector import dual_detector_falcon_2
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm

def load_small_data(fp: str = "train_small.csv"):
    """
    Load the train_small.csv file into memory from the given filepath.
    Returns a DataFrame.
    """
    fp = Path(fp)
    
    if not fp.exists():
        raise FileNotFoundError(f"The file {fp} does not exist.")
    
    return pd.read_csv(fp)

from raid import run_detection, run_evaluation
from DualDetector import dual_detector_falcon_2


print("loading detector")
detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=True)

# Define your detector function
def my_detector(texts: list[str]) -> list[float]:
    scores = detector.compute_score(texts)
    return [1 - score for score in scores]

# custom version for testing
print("loading training set")
train_df = load_small_data("train_smallester.csv")

print("running detector on training data")
# Run your detector on the dataset

# def run_detection(f, df):
#     # Make a copy of the IDs of the original dataframe to avoid editing in place
#     scores_df = df[["id"]].copy()

#     # Run the detector function on the dataset and put output in score column
#     scores_df["score"] = f(df["generation"].tolist())

#     # Convert scores and ids to dict in 'records' format for seralization
#     # e.g. [{'id':'...', 'score':0}, {'id':'...', 'score':1}, ...]
#     results = scores_df[["id", "score"]].to_dict(orient="records")

#     return results

def run_detection(f, df, batch_size=8):
    scores_df = df[["id"]].copy()
    all_scores = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["generation"].iloc[i:i+batch_size].tolist()
        try:
            batch_scores = f(batch)
            if not isinstance(batch_scores, list):
                print(f"Warning: Expected a list from f(batch), got {type(batch_scores)}")
                batch_scores = list(batch_scores)
            all_scores.extend(batch_scores)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            print(f"Problematic batch: {batch}")
            raise  # Re-raise the exception after printing debug info

    if len(all_scores) != len(scores_df):
        print(f"Warning: Number of scores ({len(all_scores)}) doesn't match number of rows ({len(scores_df)})")

    scores_df["score"] = all_scores
    results = scores_df[["id", "score"]].to_dict(orient="records")
    return results

predictions = run_detection(my_detector, train_df, batch_size=8)

print("evaluating detector's performance")
# Evaluate your detector predictions
evaluation_result = run_evaluation(predictions, train_df)

print("Evaluation Results:")
print(evaluation_result)

# Initialize your detector
# device = "cuda" if torch.cuda.is_available() else "cpu"
# detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=True, device=device)

# def my_detector(texts: list[str]) -> list[float]:
#     scores = detector.compute_score(texts)
#     return [1 - score for score in scores]

# def process_in_chunks(df, chunk_size=100):
#     all_predictions = []
#     for i in range(0, len(df), chunk_size):
#         chunk = df.iloc[i:i+chunk_size]
#         predictions = run_detection(my_detector, chunk)
#         all_predictions.extend(predictions)
#         print(f"Processed chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}")
#     return all_predictions

# # Load the data
# train_df = pd.read_csv("train_smallest.csv", engine='python')

# try:
#     # Process the data in chunks
#     chunk_size = 100  # Adjust this based on your available memory
#     predictions = process_in_chunks(train_df, chunk_size)

#     # Evaluate your detector predictions
#     evaluation_result = run_evaluation(predictions, train_df)

#     # Print the evaluation results
#     print("Evaluation Results:")
#     print(evaluation_result)

# except Exception as e:
#     print(f"An error occurred: {str(e)}")
    
#     if isinstance(e, torch.cuda.OutOfMemoryError):
#         print("CUDA out of memory error. Trying to process on CPU...")
#         detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=False, device="cpu")
#         predictions = process_in_chunks(train_df, chunk_size)
#         evaluation_result = run_evaluation(predictions, train_df)
#         print("Evaluation Results (CPU):")
#         print(evaluation_result)