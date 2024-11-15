## Introduction

This document describes the installation and usage instructions of the developed model as well as the methodology and achieved results.

### Instructions
**install.sh** installs the dependencies and a local virtual environment to run the training and evaluation scripts.

**train/train.sh** starts training using *data.json* located at the project level.

**run.sh** runs the streamlit application either in the same virtual environment if called with **--local** option, otherwise in a docker container.

### Method
I have decided to fine-tune *google/flan-t5-small* model from huggingface for this task. Before fine-tuning I tried to prompt with the examples given in the assignment, but the outputs were far from being correct. I also tried to fine-tune stronger models such as *flan-t5-base* and *codegemma*, but my 4 Gig gpu only allowed to fine-tune the simplest *google/flan-t5-small* model that has 80M parameters. This model is fast for inference, which is one of the requirements.

I asked chatgpt to generate examples (prompt-output pairs) by using all available nodes, similar to the example prompts given in the assignment description. That way, I obtained the *data.json* file found in the repository. The file contains 169 examples. This was enough to fine-tune the model.

The data was split into train and test partions with 80/20 ratio. The model was trained for a maximum of 100 epochs. Early stopping ensured the model did not overfit. AdamW optimizer and cosine learning rate scheduler with warmup were used. Small learning rate maintained stable learning and weight decay helped with regularization. The model was checkpointed according to the validation loss obtained after each epoch.

I employed BLEU score (Bilingual Evaluation Understudy Score) and sequence-level accuracy (exact match) for performance evaluation. Blue score measures n-gram overlap between predicted and reference sequences. Our model showed perfect BLEU score, 100.0, on our validation set (34 samples). Accuracy counts a prediction as true when both the order and value of the generated nodes matches the ground truth. With that, the trained model achieved 0.294 score on the validation set, where a perfect score is 1.0. According to this metric, the performance is far from being perfect. Because, it does not require exact match between the predictions and the ground truth. Nevertheless, fine-tuning a 7b parameter model such as *codegemma* should yield greater performance and still maintain relatively good inference speed. However, I was not able to do that in this assignment due to resource limitations. Also, increasing the amount of fine tuning data and applying data augmentation techniques such as changing the order of tasks in the prompt and the output accordingly, or rewriting the prompt while maintaining the meaning and the output should increase the generalization and robustness of the model.

### Unit tests
All five unit tests in *tests/test_train.py* passed once the test was run with pytest as *pytest tests/* at the project root.

### Notes
After receiving *remote: error: File train/finetuned_model/model.safetensors is 293.60 MB; this exceeds GitHub's file size limit of 100.00 MB* error, I have ignored the *train/finetuned_model/model.safetensors* pushing to the repository. Therefore, the model needs to be retrained to be able to test *app.py* after cloning this repo.
