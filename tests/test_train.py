import pytest
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict
from train.train import tokenize_function, compute_metrics
from typing import Dict, List

# Load tokenizer for testing tokenization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Mock data similar to the examples in data.json
mock_data: List[Dict[str, str]] = [
    {"prompt": "Navigate to a new page after a delay of 3 seconds when the user clicks a button.", "output": "[OnClick] [Delay] [Navigate]"},
    {"prompt": "Fetch user data and display it in a modal when a button is clicked.", "output": "[OnClick] [FetchData] [DisplayModal]"}
]

# Allowed nodes list
ALLOWED_NODES: List[str] = [
    "[OnVariableChange]", "[OnKeyRelease]", "[OnKeyPress]", "[OnClick]", "[OnWindowResize]", "[OnMouseEnter]", 
    "[OnMouseLeave]", "[OnTimer]", "[Console]", "[Alert]", "[Log]", "[Assign]", "[Delay]", "[SendRequest]", "[Navigate]", 
    "[Save]", "[Delete]", "[PlaySound]", "[PauseSound]", "[StopSound]", "[Branch]", "[Map]", "[Filter]", "[Reduce]", 
    "[Sort]", "[GroupBy]", "[Merge]", "[Split]", "[Show]", "[Hide]", "[Update]", "[DisplayModal]", "[CloseModal]", 
    "[Highlight]", "[Tooltip]", "[RenderChart]", "[FetchData]", "[StoreData]", "[UpdateData]", "[DeleteData]", 
    "[CacheData]"
]

def test_data_loading() -> None:
    """
    Test that data is correctly loaded and accessible.
    Ensures data.json is not empty.
    """
    with open("data.json", "r") as file:
        data = json.load(file)
    assert len(data) > 0, "The data.json file is empty."

def test_tokenize_function() -> None:
    """
    Test the tokenization function to ensure both inputs and labels are tokenized correctly.
    Checks that output includes 'input_ids' and 'labels' with non-zero length.
    """
    result = tokenize_function(mock_data[0])
    assert "input_ids" in result, "Missing 'input_ids' in tokenization output."
    assert "labels" in result, "Missing 'labels' in tokenization output."
    assert len(result["input_ids"]) > 0, "'input_ids' is empty after tokenization."
    assert len(result["labels"]) > 0, "'labels' is empty after tokenization."

def test_compute_metrics() -> None:
    """
    Test that compute_metrics calculates accuracy correctly.
    Uses a mock set of predictions and labels for validation.
    """
    predictions = np.array([[0, 1, 2, 3, 4]])
    labels = np.array([[0, 1, 2, 3, 4]])
    eval_preds = (predictions, labels)

    result = compute_metrics(eval_preds)
    assert "accuracy" in result, "Missing 'accuracy' in computed metrics."
    assert result["accuracy"] == 1.0, "Accuracy calculation is incorrect."

def test_tokenize_function_output_structure() -> None:
    """
    Verify that the tokenization output includes the required keys.
    Ensures 'input_ids', 'attention_mask', and 'labels' are present.
    """
    tokenized = tokenize_function(mock_data[0])
    assert "input_ids" in tokenized, "Tokenization output missing 'input_ids'."
    assert "attention_mask" in tokenized, "Tokenization output missing 'attention_mask'."
    assert "labels" in tokenized, "Tokenization output missing 'labels'."

def test_output_elements_are_valid() -> None:
    """
    Verify that all elements in the 'output' field of data.json are valid nodes.
    Checks each token in 'output' against the ALLOWED_NODES list.
    """
    with open("data.json", "r") as file:
        data = json.load(file)
    
    for entry in data:
        outputs = entry["output"].split()
        invalid_nodes = [node for node in outputs if node not in ALLOWED_NODES]
        assert not invalid_nodes, f"Invalid nodes found: {invalid_nodes}"
