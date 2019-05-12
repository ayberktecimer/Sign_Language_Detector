# Sign Language Translator
Sign Language Translator is a machine learning project aims to translate hand gestures to strings. We developed this as a term project of [CS 464 Machine Learning](http://ciceklab.cs.bilkent.edu.tr/ercumentcicek/cs-464-introduction-to-machine-learning-spring-2019/) course.
## How to use
1) Clone the repo: `git clone https://github.com/ayberktecimer/Sign_Language_Detector.git && cd Sign_Language_Detector `
2) (Optional) Create a virtual environment
3) Install dependencies: `pip3 install -r requirements.txt `
4) Train models: `cd src && python3 Runner.py ` Models are saved in `generatedModels` directory

## Dataset
https://www.kaggle.com/emresulun/sign-language-letter-images

Context Sign language letters captured by webcam and converted to grayscale images.<br />
Letters are not exactly ASL (American Sign Language) but they are adapted from ASL.

Content <br />
2125 images <br />
Image size is 300x300 <br />
The first letter indicates the label (e.g. A-1554389290.JPG) <br />
"J" is missing
