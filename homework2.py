# Richard Yang
# COSC 4570 - Data Mining
# University of Wyoming - Spring 2017
# Homework 2: Document Similarity & Hashing

import argparse
import os
import string
import time
import random
import binascii
import nltk
from nltk import ngrams
from decimal import *
import numpy as np

# Path to NLTK models
dest = os.getcwd()+"/nltk"
nltk.data.path.append(dest)

'''
Download models from NLTK
'''
def download():
	# Sentence tokenizer
	nltk.download('punkt', download_dir=dest)

'''
Sanitizes a document by getting its tokens and joining them together.
Adapted from:
http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
'''
def sanitize(document):
	# Make all characters lower case
	filter_lower = document.lower()

	# Remove all the unicode characters if found
	filter_unicode = filter_lower.decode('unicode_escape').encode('ascii','ignore')

	# Remove all punctuation
	filter_punc = filter_unicode.translate(None, string.punctuation)

	# Tokenize using NLTK
	tokens = nltk.word_tokenize(filter_punc)
	sanitized = ' '.join(tokens)
	return sanitized

'''
Return n-grams based on characters
Adapted from:
http://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-using-python
'''
def ngram_c(document, n):
	grams = [document[i:i+n] for i in range(len(document)-n+1)]
	
	# Remove duplicate ngrams while preserving order
	unique = set()
	unique_add = unique.add
	return [x for x in grams if not (x in unique or unique_add(x))]

'''
Return n-grams based on words
'''
def ngram_w(document, n):
	grams = ngrams(document.split(), n)

	# Remove duplicate ngrams while preserving order
	unique = set()
	unique_add = unique.add
	return [x for x in grams if not (x in unique or unique_add(x))]

'''
Generate a list of random coefficients for the random hash functions depending on rhc,
while ensuring that the same value does not appear multiple times in the list.
Adapted from in-class IPYNB
'''
def pickRandomCoeffs(rhc):
    # Create a list of 'k' random values.
    randList = []
  
    while rhc > 0:
    # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID) 
 
    # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID) 
    
    # Add the random number to the list.
        randList.append(randIndex)
        rhc = rhc - 1
    
    return randList

'''
Main function
'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generates n-grams from the given documents.')
	parser.add_argument('-d', "--download", help='Download the NLTK models.', action="store_true")
	parser.add_argument('-v', "--verbose", help='Performs the FULL experiment for finding ideal RHC.', action="store_true")
	args = parser.parse_args()
	if args.download:
		download()

	cwd = os.getcwd()+'/data/'

	sh1, sh2, sh3 = ({} for i in range(3))
	for name in ['D1','D2','D3','D4']:
		with open(cwd+name+'.txt', 'r') as raw_data:
			for line in raw_data:
				cleaned = sanitize(line)
				sh1[name] = ngram_c(cleaned,2) # 2-gram based on char
				sh2[name] = ngram_c(cleaned,3) # 3-gram based on char
				sh3[name] = ngram_w(cleaned,2) # 2-gram based on word

	'''
	Makes a table for Q1 and Q2: 
	How many distinct k-grams/shingles are there for each document with each type
	of k-gram/shingle?
	Compute the Jaccard Similarity between all pairs of documents for each type of
	k-gram.
	'''			
	# Number of distinct k-grams for each doc and type
	data_matrix = [['','D1', 'D2', 'D3', 'D4'],
	               ['SH1', len(sh1['D1']), len(sh1['D2']), len(sh1['D3']), len(sh1['D4'])],
	               ['SH2', len(sh2['D1']), len(sh2['D2']), len(sh2['D3']), len(sh2['D4'])],
	               ['SH3', len(sh3['D1']), len(sh3['D2']), len(sh3['D3']), len(sh3['D4'])]]
	
	print("-----Number of distinct shingles per doc-----")
	s = [[str(e) for e in row] for row in data_matrix]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print '\n'.join(table)

	
	jaccard_matrix = {}
	# Compute Jaccard similarity between all docs and types
	for i in range (1,5):
		for j in range (i+1,5):
			# Numerator of Jaccard calculation
			jaccard_1n = float(len(set(sh1['D'+str(i)]) & set(sh1['D'+str(j)])))
			# Denominator of Jaccard calculation
			jaccard_1d = float(len(set(sh1['D'+str(i)]) | set(sh1['D'+str(j)])))
			jaccard_1 = "%0.5f"%float(jaccard_1n/jaccard_1d)

			jaccard_2n = float(len(set(sh2['D'+str(i)]) & set(sh2['D'+str(j)])))
			jaccard_2d = float(len(set(sh2['D'+str(i)]) | set(sh2['D'+str(j)])))
			jaccard_2 = "%0.5f"%float(jaccard_2n/jaccard_2d)
			
			jaccard_3n = float(len(set(sh3['D'+str(i)]) & set(sh3['D'+str(j)])))
			jaccard_3d = float(len(set(sh3['D'+str(i)]) | set(sh3['D'+str(j)])))
			jaccard_3 = "%0.5f"%float(jaccard_3n/jaccard_3d)

			jaccard_matrix[(i,j)] = [jaccard_1, jaccard_2, jaccard_3]
	
	jm = jaccard_matrix

	data_matrix = [['','D1-D2', 'D1-D3', 'D1-D4', 'D2-D3', 'D2-D4', 'D3-D4'],
	               ['SH1', jm[(1,2)][0], jm[(1,3)][0], jm[(1,4)][0], jm[(2,3)][0], jm[(2,4)][0], jm[(3,4)][0]],
	               ['SH2', jm[(1,2)][1], jm[(1,3)][1], jm[(1,4)][1], jm[(2,3)][1], jm[(2,4)][1], jm[(3,4)][1]],
	               ['SH3', jm[(1,2)][2], jm[(1,3)][2], jm[(1,4)][2], jm[(2,3)][2], jm[(2,4)][2], jm[(3,4)][2]]]

	print("\n\n-----Jaccard Similarity Matrix-----")
	s = [[str(e) for e in row] for row in data_matrix]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print '\n'.join(table)

	# Generate min-hash signatures for D1 and D2
	sh2_hashes = {}
	sh2_hashes['D1'], sh2_hashes['D2'] = ([] for i in range(2))
	for name in ['D1','D2']:
		for gram in sh2[name]:
			hashgram = binascii.crc32(gram) & 0xffffffff
			#print gram + "," + str(hashgram)
			sh2_hashes[name].append(hashgram)

	'''
	Build min-hash signatures for D1 and D2 using different number of rhc
	Adapted from in-class IPYNB
	'''
	rhc_list = [10,20,60,200,500]
	hash_time = {}
	compare_time = {}
	normalized_L0 = {}
	for rhc in rhc_list:
		# Time this step.
		hash_t0 = time.time()

		maxShingleID = 2**32-1
		nextPrime = 4294967311
		coeffA = pickRandomCoeffs(rhc)
		coeffB = pickRandomCoeffs(rhc)
		signatures = []

		# For each document...
		for name in ['D1','D2']:
		    shingleIDSet = sh2_hashes[name]
		    signature = []
		    # For each of the random hash functions...
		    for i in range(0, rhc):
		        # For each of the shingles actually in the document, calculate its hash code
		        # using hash function 'i'. 
		        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
		        # the maximum possible value output by the hash.
		        minHashCode = nextPrime + 1
		    
		        # For each shingle in the document...
		        for shingleID in shingleIDSet:
		            # Evaluate the hash function.
		            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
		      
		            # Track the lowest hash code seen.
		            if hashCode < minHashCode:
		                minHashCode = hashCode

		        # Add the smallest hash code value as component number 'i' of the signature.
		        signature.append(minHashCode)

		    # Store the MinHash signature for this document.
		    signatures.append(signature)

		# Calculate the elapsed time (in seconds)
		hash_time[rhc] = (time.time() - hash_t0)

		# Time this step.
		comp_t0 = time.time()
		# Get the MinHash signature for document 1.
		signature1 = signatures[0]
		# Get the MinHash signature for document 2.
		signature2 = signatures[1]

		count = 0
        # Count the number of positions in the minhash signature which are equal.
		for k in range(0, rhc):
			#print(str(signature1[k])+","+str(signature2[k]))
			count = count + (signature1[k] == signature2[k])
    
 	    # Record the percentage of positions which matched.    
		normalized_L0[rhc] = (float(count) / float(rhc))
		compare_time[rhc] = (time.time() - comp_t0)

	ht = hash_time
	ct = compare_time
	l0 = normalized_L0
	data_matrix = [['RHC','Hash Time (s)', 'Compare Time (s)', 'L0 Norm']]
	for rhc in rhc_list:
		data_matrix.append([rhc, ht[rhc], ct[rhc], l0[rhc]])

	print("\n\n-----RHC Comparison-----")
	s = [[str(e) for e in row] for row in data_matrix]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print '\n'.join(table)

	if args.verbose:
		hash_time, compare_time, normalized_L0 = ({} for i in range(3))
		# NP Range of 10->100, 100->1000, 1000->10000
		rhc_list = list(np.arange(10,100,10)) + list(np.arange(100,1000,100)) + list(np.arange(1000,10000,1000))
		for rhc in rhc_list:
			hash_time[rhc] = 0
			compare_time[rhc] = 0
			normalized_L0[rhc] = 0

		# Since the L0 norm changes every time, we perform the experiment 5 times and take the average
		for i in range(5):
			for rhc in rhc_list:
				hash_t0 = time.time()
				maxShingleID = 2**32-1
				nextPrime = 4294967311
				coeffA = pickRandomCoeffs(rhc)
				coeffB = pickRandomCoeffs(rhc)
				signatures = []
				for name in ['D1','D2']:
				    shingleIDSet = sh2_hashes[name]
				    signature = []
				    for i in range(0, rhc):
				        minHashCode = nextPrime + 1
				        for shingleID in shingleIDSet:
				            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
				            if hashCode < minHashCode:
				                minHashCode = hashCode
				        signature.append(minHashCode)
				    signatures.append(signature)

				hash_time[rhc] = hash_time[rhc] + (time.time() - hash_t0)

				comp_t0 = time.time()
				signature1 = signatures[0]
				signature2 = signatures[1]
				count = 0
				for k in range(0, rhc):
					count = count + (signature1[k] == signature2[k])

				normalized_L0[rhc] = normalized_L0[rhc] + (float(count) / float(rhc))
				compare_time[rhc] = compare_time[rhc] + (time.time() - comp_t0)

		for rhc in rhc_list:
			hash_time[rhc] = float(hash_time[rhc])/float(5)
			compare_time[rhc] = float(compare_time[rhc])/float(5)
			normalized_L0[rhc] = float(normalized_L0[rhc])/float(5)

		ht = hash_time
		ct = compare_time
		l0 = normalized_L0
		data_matrix = [['RHC','Hash Time (s)', 'Compare Time (s)', 'L0 Norm']]
		for rhc in rhc_list:
			data_matrix.append([rhc, ht[rhc], ct[rhc], l0[rhc]])

		print("\n\n-----RHC Comparison-----")
		s = [[str(e) for e in row] for row in data_matrix]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print '\n'.join(table)

