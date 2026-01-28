# Sentiment Analysis on Stanford Sentiment Treebank (SST)

This repository provides a comprehensive framework for performing sentiment analysis on the **Stanford Sentiment Treebank (SST)**. The project implements and compares multiple neural network architectures, ranging from baseline **Log-Linear** models to advanced **Long Short-Term Memory (LSTM)** networks, to classify the sentiment of movie reviews into positive, negative, or neutral categories.

## Project Overview

The core objective of this project is to evaluate how different text representation methods impact the performance of sentiment classification. By utilizing the hierarchical structure of the SST dataset, the implementation allows for training on both full sentences and their constituent sub-phrases. The framework supports three primary data representation strategies: **One-Hot Averaging**, **Word2Vec Averaging**, and **Word2Vec Sequences**. These methods provide a progression from simple frequency-based representations to dense, context-aware embeddings suitable for recurrent architectures.

## Repository Structure

The project is organized into modular components to separate data handling from model implementation and training logic.

| File / Directory | Description |
| :--- | :--- |
| `exercise_blanks.py` | The primary script containing model definitions, training loops, and evaluation metrics. |
| `data_loader.py` | A utility module responsible for parsing the SST dataset and managing sentiment tree structures. |
| `stanfordSentimentTreebank/` | The target directory for the raw SST dataset files, including sentences, splits, and labels. |
| `README.md` | Documentation providing an overview of the project, its structure, and usage instructions. |

## Technical Implementation

The implementation leverages **PyTorch** for model development and **Gensim** for accessing pre-trained word embeddings. The `DataManager` class serves as the central hub for data operations, abstracting the complexities of batching and sequence padding.

### Model Architectures

The framework includes two distinct model classes designed for different input types. The **Log-Linear** model is a straightforward linear classifier optimized for averaged embeddings, while the **LSTM** model is a recurrent network designed to process sequences of word vectors, capturing the temporal dependencies inherent in natural language.

### Data Utilities

The `data_loader.py` script handles the intricate task of building `SentimentTreeNode` objects from the raw text files. This allows the model to access the sentiment values of every node in the parse tree, enabling a more granular training process. The dataset is automatically split into training, validation, and test sets according to standard SST ratios.

## Usage Instructions

To utilize this framework, ensure that the `stanfordSentimentTreebank` directory is populated with the necessary dataset files, such as `datasetSentences.txt` and `sentiment_labels.txt`. The main execution script, `exercise_blanks.py`, can be run directly to initiate the training and evaluation process. It will automatically handle the downloading of the `word2vec-google-news-300` model if it is not already present in the local cache.

## Authors

This project was developed by **Malak Laham** and **Zenab Waked** as part of their studies in Natural Language Processing.
