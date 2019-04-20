from src.Train import trainLDA, trainSVM
from src.Visualization import plotConfusionMatrix

# Train
ldaParameters = {
	"n_components": 1
}
trainLDA(ldaParameters)
trainSVM()

# Test
from src.Test import testLDA, testSVM

plotConfusionMatrix(*testLDA())
plotConfusionMatrix(*testSVM())
