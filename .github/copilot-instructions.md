# General
This repo is used for a YouTube video about Kaggle Playground Competition season 5 episode 8. Binary Classification with a Bank Dataset.

Your Goal: Your goal is to predict whether a client will subscribe to a bank term deposit.

## Dataset Description
The competition dataset (train and test) was generated using a deep learning model
trained on the Bank Marketing Dataset. Feature distributions are similar, but not
identical, to the original. You may use the original dataset to explore differences
or to see if including it in training improves model performance.

## Files
data/train.csv - the training dataset; y is the binary target
data/test.csv - the test dataset; your objective is to predict the probability y for each row
data/sample_submission.csv - a sample submission file in the correct format

## Guidelines
uv is used. Do not give suggestions about requirements.txt or other outdated python ways.

Ruff is used for linting and formatting. See .github/workflows/ruff.yml for workflow.

Latest version of any dependencies is desired. Version pinned, do not pin using keyword latest.

Code should be simple rather than complex. Creating tests is expected, but not mandatory.

I don't have a GPU, only a weak CPU. Keep it light.
