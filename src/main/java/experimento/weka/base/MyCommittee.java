package experimento.weka.base;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public abstract class MyCommittee extends MyClassifier{
	public abstract Evaluation evaluateClassifier(int folds,MyClassifier myclassifier,int quantClassifiers, int seed, Instances data) throws Exception;
}
