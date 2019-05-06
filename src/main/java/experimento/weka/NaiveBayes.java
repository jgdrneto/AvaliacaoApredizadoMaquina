package experimento.weka;

import experimento.weka.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class NaiveBayes extends MLClassifier{

	weka.classifiers.bayes.NaiveBayes classifier;
	
	public NaiveBayes() {
		this.classifier = new weka.classifiers.bayes.NaiveBayes();
	}
	
	public Evaluation evaluateClassifier(int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
		
	}
	
	@Override
	public String toCSVHeader() {
		return "FOLDS,SEED,DATA,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED\n\n";
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		
		String csv =seed + "," + 
					folds + "," +
					dataValue.name() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
		
	}
}
