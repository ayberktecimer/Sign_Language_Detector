from src.Train import trainLDA, trainSVM
from src.Test import testLDA, testSVM
from src.Visualization import plotConfusionMatrix

# Train
ldaParameters = {
	"n_components": 1
}
trainLDA(ldaParameters)
trainSVM()

# Test
actual, predicted = testLDA()
plotConfusionMatrix(actual, predicted)
testSVM()
