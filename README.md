# Optimal Converse
Optimal Converse is a proof-of-concept tool implemented in Python, designed to measure the distribution of communication dimensions in transcribed conversations related to collaborative problem-solving.

## Content
This repository contains the necessary resources to replicate and run the communication dimension analyzer tool, Optimal Converse. The repository includes the following folders:

### Data:
This folder contains a CSV file named **"combined.csv"**, that includes labeled transcriptions from [Does distance still matter? revisiting collaborative distributed software design](https://ieeexplore.ieee.org/document/8409905)

### Python:
This folder contains the source code and implementation of the collaborative communication dimension analyzer tool, Optimal Converse. The files in the folder are:

- **graph.py**  
Loads the three communication dimensions "Conversation Management", "Active Discussion", and "Creative Conflict" and generates
a graph based on the provided data.

- **lemma_experiment.py**  
Performs an experiment to evaluate the performance of a Random Forest classifier with lemmatized text data.

- **stats.py**  
Performs statistical analysis on a dataset containing transcribed text data from **"data/combined.csv"**.
This script provides insights into the performance of different classifiers, vectorizers, and samplers in analyzing transcribed text data,
using metrics such as accuracy, precision, and recall.

## Required dependencies
To successfully run Optimal Converse, ensure that the following dependencies are installed:

- scikit-learn
- pandas
- numpy
- matplotlib
- imblearn
- nltk

You can install these dependencies using a package manager like pip. Open a terminal or command prompt and execute the following commands:

pip install scikit-learn  
pip install pandas  
pip install numpy  
pip install matplotlib
pip install imblearn
pip install nltk

## Run with Python
In order to get Optimal Converse running, follow these steps:

1. Clone the repository to you local machine:  
   git clone git@github.com:ProgrammerMarcus/optimal-converse.git
2. Navigate to the repository on your machine:  
   cd repo-name
3. [next-step]
