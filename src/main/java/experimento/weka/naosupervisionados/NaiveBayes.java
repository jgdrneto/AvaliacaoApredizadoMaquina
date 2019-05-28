package experimento.weka.naosupervisionados;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class NaiveBayes extends MyClassifier{

	weka.classifiers.bayes.NaiveBayes classifier;
	
	public NaiveBayes() {
		this.classifier = new weka.classifiers.bayes.NaiveBayes();
	}
	
	public void buildClassifier(Instances data) throws Exception {

		super.buildClassifier(classifier,data);
		
	}
	
	@Override
	public double classifyInstance(Instance i) throws Exception {
		return this.classifier.classifyInstance(i);
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
}
