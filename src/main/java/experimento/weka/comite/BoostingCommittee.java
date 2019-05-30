package experimento.weka.comite;

import java.util.List;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class BoostingCommittee extends MyCommittee{
	
	AdaBoostM1 classifier;
	
	public BoostingCommittee() {
		this.classifier = new AdaBoostM1();
	}
	
	private BoostingCommittee(BoostingCommittee nClassifier) throws Exception {
		this.classifier = new AdaBoostM1();
		this.quant = nClassifier.getQuant();
		this.cClass = nClassifier.getcClass();
		
		AdaBoostM1 clsAnt = (AdaBoostM1)nClassifier.getClassifier();
		
		this.classifier.setClassifier(AbstractClassifier.makeCopy(nClassifier.getClassifier()));
		this.classifier.setNumIterations(clsAnt.getNumIterations());
		this.classifier.setSeed(clsAnt.getSeed());
		this.classifier.setNumDecimalPlaces(clsAnt.getNumDecimalPlaces());
	}
	
	public Evaluation evaluateClassifier(int folds,List<MyClassifier> myclassifiers,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception{
		
		if(!myclassifiers.isEmpty()) {
			
			this.quant= quantClassifiers;
			this.cClass = myclassifiers.get(0).getClass();
			
			this.classifier.setClassifier(myclassifiers.get(0).getClassifier());
			this.classifier.setNumIterations(quantClassifiers);
			this.classifier.setSeed(seed);
			this.classifier.setNumDecimalPlaces(4);
			
			return evaluateClassifier(folds, this.classifier, seed, datavalue, data);
		
		}else {
			throw new RuntimeException("Lista de classificadores vazia");
		}
	}
	@Override
	public Classifier getClassifier() {
		return this.classifier;
	}
	
	@Override
	public MyClassifier copy() throws Exception {
		return new BoostingCommittee(this);
	}
	
}
