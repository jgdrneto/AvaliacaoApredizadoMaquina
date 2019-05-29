package experimento.weka.comite;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Instances;

public class BaggingCommittee extends MyCommittee{
	
	Bagging classifier;
	
	public BaggingCommittee() {
		this.classifier = new Bagging();
	}
	
	public Evaluation evaluateClassifier(int folds,MyClassifier myclassifier,int quantClassifiers, int seed, Instances data) throws Exception{
		
		this.classifier.setClassifier(myclassifier.getClassifier());
		this.classifier.setNumIterations(quantClassifiers);
		this.classifier.setSeed(seed);
		
		return evaluateClassifier(folds, this.classifier, seed, DATAVALUE.NOT_NORMALIZED, data);
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

	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}

	@Override
	public double classifyInstance(Instance i) throws Exception {
		
		return this.classifier.classifyInstance(i);

	}
	
}
