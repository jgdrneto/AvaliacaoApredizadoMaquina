package experimento.weka.comite;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.core.Instances;

public class HeterogeneousStackingCommittee extends MyCommittee{
	
	Stacking classifier;
	
	public HeterogeneousStackingCommittee() {
		this.classifier = new Stacking();
	}
	
	public Evaluation evaluateClassifier(int folds,List<MyClassifier> myclassifiers,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception{
		
		this.quant=myclassifiers.size();
		this.cClass = myclassifiers.get(0).getClass();
		
		Classifier[] classifiers = new Classifier[myclassifiers.size()];
		
		for(int i=0;i<myclassifiers.size();i++) {
			classifiers[i] = myclassifiers.get(i).getClassifier();
		}

		this.classifier.setClassifiers(classifiers);
		this.classifier.setSeed(seed);
		this.classifier.setNumDecimalPlaces(4);
		this.classifier.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
		
		return evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}
	
	@Override
	public MyClassifier copy() throws Exception {
		return new HeterogeneousStackingCommittee();
	}
	
}
