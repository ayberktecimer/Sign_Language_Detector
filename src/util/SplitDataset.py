"""
File: SplitDataset
Author: Emre Sülün
Date: 20.04.2019
Project: Sign_Language_Detector
Description: Splits images into train and test folders. Split ratio is determined by TRAIN_SIZE variable
"""
import os
import shutil
import string

source = '../../samples'
trainDestination = '../../samples/train'
testDestination = '../../samples/test'
TRAIN_SIZE = 50

# Firstly,  merge all files into samples folder
oldTrainFiles = os.listdir(trainDestination)
for f in oldTrainFiles:
	f = trainDestination + "/" + f
	shutil.move(f, source)

oldTestFiles = os.listdir(testDestination)
for f in oldTestFiles:
	f = testDestination + "/" + f
	shutil.move(f, source)

# Then, distribute files into train and test folders
chars = {}
for char in string.ascii_uppercase:
	chars[char] = TRAIN_SIZE

files = os.listdir(source)

for f in files:
	letter = f[0]
	if letter == 't':  # Skip test and train folders
		continue
	if chars[letter] > 0:
		f = source + "/" + f
		shutil.move(f, trainDestination)
		chars[letter] = chars[letter] - 1


files = os.listdir(source)
for f in files:
	letter = f[0]
	if letter == 't':  # Skip test and train folders
		continue
	f = source + "/" + f
	shutil.move(f, testDestination)
