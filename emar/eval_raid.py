from raid import run_detection, run_evaluation
from raid.utils import load_data
from DualDetector import dual_detector_falcon_2
import pandas as pd
from pathlib import Path
import torch

# def load_small_data(fp: str = "train_small.csv"):
#     """
#     Load the train_small.csv file into memory from the given filepath.
#     Returns a DataFrame.
#     """
#     fp = Path(fp)
    
#     if not fp.exists():
#         raise FileNotFoundError(f"The file {fp} does not exist.")
    
#     return pd.read_csv(fp)

# from raid import run_detection, run_evaluation
# from DualDetector import dual_detector_falcon_2


# detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=True)

# # Define your detector function
# def my_detector(texts: list[str]) -> list[float]:
#     scores = detector.compute_score(texts)
#     return [1 - score for score in scores]

# # custom version for testing
# train_df = load_small_data("train_smallest.csv")

# # Run your detector on the dataset
# predictions = run_detection(my_detector, train_df)

# # Evaluate your detector predictions
# evaluation_result = run_evaluation(predictions, train_df)

# print("Evaluation Results:")
# print(evaluation_result)

# Initialize your detector
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=True, device=device)

def my_detector(texts: list[str]) -> list[float]:
    scores = detector.compute_score(texts)
    return [1 - score for score in scores]

def process_in_chunks(df, chunk_size=100):
    all_predictions = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        predictions = run_detection(my_detector, chunk)
        all_predictions.extend(predictions)
        print(f"Processed chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}")
    return all_predictions

# Load the data
train_df = pd.read_csv("train_smallest.csv", engine='python')

try:
    # Process the data in chunks
    chunk_size = 100  # Adjust this based on your available memory
    predictions = process_in_chunks(train_df, chunk_size)

    # Evaluate your detector predictions
    evaluation_result = run_evaluation(predictions, train_df)

    # Print the evaluation results
    print("Evaluation Results:")
    print(evaluation_result)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    
    if isinstance(e, torch.cuda.OutOfMemoryError):
        print("CUDA out of memory error. Trying to process on CPU...")
        detector = dual_detector_falcon_2.DualDetector_2(use_bfloat16=False, device="cpu")
        predictions = process_in_chunks(train_df, chunk_size)
        evaluation_result = run_evaluation(predictions, train_df)
        print("Evaluation Results (CPU):")
        print(evaluation_result)