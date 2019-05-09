package experimento.weka;


import experimento.weka.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class ArtificialNeuralNetwork extends MLClassifier{

	MultilayerPerceptron classifier;
	
	public ArtificialNeuralNetwork() {
		this.classifier = new MultilayerPerceptron();
		this.classifier.setMomentum(0.8);
	}
	
	public Evaluation evaluateClassifier(int ciclos,int neuroniosEscondidos,double taxaAprendizado,int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		this.classifier.setTrainingTime(ciclos);
		this.classifier.setHiddenLayers(String.valueOf(neuroniosEscondidos));
		this.classifier.setLearningRate(taxaAprendizado);
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public String toCSVHeader() {
		return "TRAINING TIME,HIDDEN LAYERS,LEARNING RATE,FOLDS,SEED,DATA,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED\n\n";
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		
		String csv =this.classifier.getTrainingTime() + "," +
					this.classifier.getHiddenLayers() + "," +
					this.classifier.getLearningRate() + "," +
					folds + "," + 
					seed + "," +
					dataValue.name() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
	}
}
