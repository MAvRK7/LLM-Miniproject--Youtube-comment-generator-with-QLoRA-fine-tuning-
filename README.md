# LLM-Miniproject--Youtube-comment-generator-with-QLoRA-fine-tuning-
This is a YouTube reply generator which uses the Mistral-7b-instruct as the base model. QLoRA has been used to fine tune this model. All the libraries and dataset used are from open sources like hugging face, the dataset used to train the reply generator is shawhin/shawgpt-youtube-comments

Introduction

The YouTube Reply Generator is designed to provide automated, contextually relevant replies to YouTube comments. It leverages the Mistral-7b-instruct model, a powerful language model, and fine-tunes it using QLoRA to optimize its performance for generating replies.

Features

Base Model: Uses the Mistral-7b-instruct model.
Fine-Tuning: Implemented using QLoRA to enhance model performance.
Dataset: Trained on the shawhin/shawgpt-youtube-comments dataset from Hugging Face.
Libraries: Utilizes open-source libraries including transformers and peft.
Installation

To get started with the project, clone the repository and install the required dependencies. The primary libraries used include transformers, peft, and datasets.

Usage

The YouTube Reply Generator can be used to generate replies to comments on YouTube. By inputting a comment, the model processes it and generates an appropriate response.

Training

The model has been fine-tuned using the QLoRA method. The training process involves:

Data Collection: Using the shawhin/shawgpt-youtube-comments dataset.
Data Tokenization: Preparing the dataset for training.
Model Preparation: Setting up the Mistral-7b-instruct model for fine-tuning.
Training Process: Implementing the QLoRA method to fine-tune the model, optimizing it for generating YouTube comment replies.
