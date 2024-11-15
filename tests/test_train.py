# tests/test_train.py
import pytest
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from train.train import tokenize_function, compute_metrics

# Load tokenizer for testing tokenization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Mock data similar to the examples in data.json
mock_data = [
    {"prompt": "Navigate to a new page after a delay of 3 seconds when the user clicks a button.", "output": "[OnClick] [Delay] [Navigate]"},
    {"prompt": "Fetch user data and display it in a modal when a button is clicked.", "output": "[OnClick] [FetchData] [DisplayModal]"}
]

# Allowed nodes list
ALLOWED_NODES = [
    "[OnVariableChange]", "[OnKeyRelease]", "[OnKeyPress]", "[OnClick]", "[OnWindowResize]", "[OnMouseEnter]", 
    "[OnMouseLeave]", "[OnTimer]", "[Console]", "[Alert]", "[Log]", "[Assign]", "[Delay]", "[SendRequest]", "[Navigate]", 
    "[Save]", "[Delete]", "[PlaySound]", "[PauseSound]", "[StopSound]", "[Branch]", "[Map]", "[Filter]", "[Reduce]", 
    "[Sort]", "[GroupBy]", "[Merge]", "[Split]", "[Show]", "[Hide]", "[Update]", "[DisplayModal]", "[CloseModal]", 
    "[Highlight]", "[Tooltip]", "[RenderChart]", "[FetchData]", "[StoreData]", "[UpdateData]", "[DeleteData]", 
    "[CacheData]"
]

def test_data_loading():
    """Test that data is correctly loaded and accessible."""
    with open("data.json", "r") as file:
        data = json.load(file)
    assert len(data) > 0  # Ensure that data.json is not empty

def test_tokenize_function():
    """Test tokenization function to ensure both inputs and labels are tokenized."""
    result = tokenize_function(mock_data[0])
    assert "input_ids" in result
    assert "labels" in result
    assert len(result["input_ids"]) > 0
    assert len(result["labels"]) > 0

def test_compute_metrics():
    """Test that compute_metrics calculates accuracy correctly."""
    predictions = np.array([[0, 1, 2, 3, 4]])
    labels = np.array([[0, 1, 2, 3, 4]])
    eval_preds = (predictions, labels)

    result = compute_metrics(eval_preds)
    assert "accuracy" in result
    assert result["accuracy"] == 1.0  # 100% accuracy in this case

def test_tokenize_function_output_structure():
    """Verify tokenization output includes necessary keys."""
    tokenized = tokenize_function(mock_data[0])
    assert "input_ids" in tokenized, "Tokenization output missing 'input_ids'"
    assert "attention_mask" in tokenized, "Tokenization output missing 'attention_mask'"
    assert "labels" in tokenized, "Tokenization output missing 'labels'"

def test_output_elements_are_valid():
    """Verify all elements in the 'output' field of data.json are valid actions."""
    with open("data.json", "r") as file:
        data = json.load(file)
    
    for entry in data:
        outputs = entry["output"].split()
        invalid_nodes = [node for node in outputs if node not in ALLOWED_NODES]
        assert not invalid_nodes, f"Invalid nodes found: {invalid_nodes}"
