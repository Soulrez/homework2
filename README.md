<snippet>
  <content>
# COSC 4570 - Homework 2
Richard Yang

COSC 4570 - Data Mining

University of Wyoming - Spring 2017

Homework 2: Document Similarity & Hashing

## Installation
An Anaconda environment profile is provided for your convenience.

`conda env create -f homework2_env.yml`

Requirements:

-Python 2.7

-nltk

-numpy



## Usage
Generates n-grams of four example documents based on characters and words. Then, calculates the Jaccard similarity between each document and type of n-gram sequencing method. Then, applies MinHashing of document 1 and document two using variable random hash counts and computes a normalized compression distance between the two documents for each RHC.
1. The code can be run directly `python homework2.py`, use -h for help messages. All output will be through the terminal.
2. (OPTIONAL) The NLTK models used are included with this repo. However, you can manually download them via the -d flag.
3. (OPTIONAL) I performed additional experiments to find the ideal RCH value. These experiments may take up a lot of RAM and CPU cycles, but they can be ran using the -v flag.


</content>
</snippet>
