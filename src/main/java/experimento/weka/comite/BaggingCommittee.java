package experimento.weka.comite;

import java.util.List;

import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;

public class BaggingCommittee extends MyCommittee{
	
	Bagging classifier;
	int percentTrainingSet;
	
	public BaggingCommittee(int nPercent) {
		this.classifier = new Bagging();
		this.percentTrainingSet = nPercent;
	}
	
	private BaggingCommittee(BaggingCommittee nClassifier) throws Exception {
		this.classifier = new Bagging();
		this.percentTrainingSet = nClassifier.getPercentTrainingSet();
		this.quant = nClassifier.getQuant();
		this.cClass = nClassifier.getcClass();
		
		Bagging clsAnt = (Bagging)nClassifier.getClassifier();
		
		this.classifier.setBagSizePercent(clsAnt.getBagSizePercent());
		this.classifier.setClassifier(AbstractClassifier.makeCopy(nClassifier.getClassifier()));
		this.classifier.setNumIterations(clsAnt.getNumIterations());
		this.classifier.setSeed(clsAnt.getSeed());
		this.classifier.setNumDecimalPlaces(clsAnt.getNumDecimalPlaces());
		this.classifier.setNumExecutionSlots(clsAnt.getNumExecutionSlots());
	}
	
	public int getPercentTrainingSet() {
		return percentTrainingSet;
	}

	public Evaluation evaluateClassifier(int folds,List<MyClassifier> myclassifiers,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception{
		
		if(!myclassifiers.isEmpty()) {
			this.quant= quantClassifiers;
			this.cClass = myclassifiers.get(0).getClass();
			
			this.classifier.setBagSizePercent(this.percentTrainingSet);
			this.classifier.setClassifier(myclassifiers.get(0).getClassifier());
			this.classifier.setNumIterations(quantClassifiers);
			this.classifier.setSeed(seed);
			this.classifier.setNumDecimalPlaces(4);
			this.classifier.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
			
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
		return new BaggingCommittee(this);
	}

}
