from src.ReportGenerator import saveTestResults
from src.Train import trainLDA, trainSVM
from src.ReportGenerator import plotConfusionMatrix


# Train
ldaParameters = {
	"n_components": 1
}
trainLDA(ldaParameters)
trainSVM()

# Test
from src.Test import testLDA, testSVM

ldaModel, ldaActual, ldaPredicted = testLDA()
svmModel, svmActual, svmPredicted = testSVM()

plotConfusionMatrix(ldaActual, ldaPredicted)
plotConfusionMatrix(svmActual, svmPredicted)
saveTestResults(ldaModel, ldaActual, ldaPredicted)
saveTestResults(svmModel, svmActual, svmPredicted)
