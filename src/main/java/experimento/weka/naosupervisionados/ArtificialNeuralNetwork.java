package experimento.weka.naosupervisionados;


import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;

public class ArtificialNeuralNetwork extends MyClassifier{

	MultilayerPerceptron classifier;
	
	public ArtificialNeuralNetwork() {
		this.classifier = new MultilayerPerceptron();
		this.classifier.setMomentum(0.8);
	}
	
	public void buildClassifier(int ciclos,int neuroniosEscondidos,double taxaAprendizado,Instances data) throws Exception {
		
		this.classifier.setTrainingTime(ciclos);
		this.classifier.setHiddenLayers(String.valueOf(neuroniosEscondidos));
		this.classifier.setLearningRate(taxaAprendizado);
		
		super.buildClassifier(classifier,data);
	}
	

	
	public void setSeed(int s) {
		this.classifier.setSeed(s);
	}
	
	public Evaluation evaluateClassifier(int ciclos,int neuroniosEscondidos,double taxaAprendizado,int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		this.classifier.setTrainingTime(ciclos);
		this.classifier.setHiddenLayers(String.valueOf(neuroniosEscondidos));
		this.classifier.setLearningRate(taxaAprendizado);
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public String toCSVHeader() {
		return "TRAINING TIME,HIDDEN LAYERS,LEARNING RATE,FOLDS,SEED,CLASSIFIER SEED,DATA,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED\n\n";
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		
		String csv =this.classifier.getTrainingTime() + "," +
					this.classifier.getHiddenLayers() + "," +
					this.classifier.getLearningRate() + "," +
					folds + "," + 
					seed + "," +
					this.classifier.getSeed() + "," +
					dataValue.name() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
	}

	@Override
	public double classifyInstance(Instance i) throws Exception {
		return this.classifier.classifyInstance(i);
	}
}
