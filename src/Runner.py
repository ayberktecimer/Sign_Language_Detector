# Train
from src.train import trainLDA, trainSVN

ldaParameters = {
	"n_components": 1
}
trainLDA(ldaParameters)
trainSVN()

# Test
from src.test import testLDA, testSVN

testLDA()
testSVN()
