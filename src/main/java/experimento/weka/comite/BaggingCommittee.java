package experimento.weka.comite;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;

public class BaggingCommittee extends MyCommittee{
	
	Bagging classifier;
	
	public BaggingCommittee() {
		this.classifier = new Bagging();
	}
	
	public Evaluation evaluateClassifier(int folds,MyClassifier myclassifier,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception{
		
		this.quant= quantClassifiers;
		this.cClass = myclassifier.getClass();
		
		this.classifier.setClassifier(myclassifier.getClassifier());
		this.classifier.setNumIterations(quantClassifiers);
		this.classifier.setSeed(seed);
		this.classifier.setNumDecimalPlaces(4);
		this.classifier.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
		
		return evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}
	
}
