from src.ReportGenerator import saveTestResults
from src.Train import trainLDA, trainSVM
from src.Test import testLDA, testSVM
from src.ReportGenerator import plotConfusionMatrix

# Train and test LDA
solverSet = ["svd", "lsqr", "eigen"]
shrinkageSet = ["auto", 0.2, 0.4, 0.6, 0.8, 1]
nSet = [5, 10, 15]

for solver in solverSet:
	for shrinkage in shrinkageSet:
		for n in nSet:
			try:
				ldaParameters = {
					"solver": solver,
					"shrinkage": shrinkage,
					"n_components": n
				}
				trainLDA(ldaParameters)
				ldaModel, ldaActual, ldaPredicted = testLDA()
				plotConfusionMatrix(ldaActual, ldaPredicted)
				saveTestResults(ldaModel, ldaActual, ldaPredicted)
			except NotImplementedError:
				pass

# Train and test SVM
cSet = [0.01, 0.1, 1, 10]
kernelSet = ["linear", "poly", "rbf", "sigmoid"]

for c in cSet:
	for kernel in kernelSet:
		try:
			svmParameters = {
				"C": c,
				"kernel": kernel,
				"gamma": "scale"
			}

			trainSVM(svmParameters)
			svmModel, svmActual, svmPredicted = testSVM()
			plotConfusionMatrix(svmActual, svmPredicted)
			saveTestResults(svmModel, svmActual, svmPredicted)
		except NotImplementedError:
			pass
