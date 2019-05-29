package experimento.weka.supervisionados;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class NaiveBayes extends MyClassifier{

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
		
		String csv =folds + "," +
					seed + "," + 
					dataValue.name() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
		
	}

	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}
}
