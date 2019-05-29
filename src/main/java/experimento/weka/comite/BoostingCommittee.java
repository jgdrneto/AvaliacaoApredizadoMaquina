package experimento.weka.comite;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class BoostingCommittee extends MyCommittee{
	
	AdaBoostM1 classifier;
	
	public BoostingCommittee() {
		this.classifier = new AdaBoostM1();
	}
	
	public Evaluation evaluateClassifier(int folds,MyClassifier myclassifier,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception{
		
		this.quant= quantClassifiers;
		this.cClass = myclassifier.getClass();
		
		this.classifier.setClassifier(myclassifier.getClassifier());
		this.classifier.setNumIterations(quantClassifiers);
		this.classifier.setSeed(seed);
		this.classifier.setNumDecimalPlaces(4);
		
		return evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}

	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}
	
}
