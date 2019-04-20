# Train
from src.train import trainLDA, trainSVM

ldaParameters = {
	"n_components": 1
}
trainLDA(ldaParameters)
trainSVM()

# Test
from src.test import testLDA, testSVM

testLDA()
testSVM()
