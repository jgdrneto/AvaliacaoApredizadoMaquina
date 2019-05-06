package experimento.weka;


import experimento.weka.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class ArtificialNeuralNetwork extends MLClassifier{

	MultilayerPerceptron classifier;
	
	public ArtificialNeuralNetwork() {
		this.classifier = new MultilayerPerceptron();
	}
	
	public Evaluation evaluateClassifier(int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public String toCSVHeader() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		// TODO Auto-generated method stub
		return null;
	}
}
