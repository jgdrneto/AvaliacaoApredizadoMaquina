package experimento.weka.naosupervisionados;


import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTree extends MyClassifier{

	J48 classifier;
	
	public DecisionTree() {
		this.classifier = new J48();
	}
	
	public void buildClassifier(Boolean unpruned,Instances data) throws Exception {
		this.classifier.setUnpruned(unpruned);
		
		super.buildClassifier(classifier,data);

	}
	
	@Override
	public double classifyInstance(Instance i) throws Exception {
		return this.classifier.classifyInstance(i);
	}
	
	public Evaluation evaluateClassifier(Boolean unpruned,int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		this.classifier.setUnpruned(unpruned);
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public String toCSVHeader() {
		
		return "UNPRUNED,FOLDS,SEED,DATA,LEAFS,SIZE,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED\n\n";
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		
		String csv =this.classifier.getUnpruned() + "," + 
					folds + "," +
					seed + "," + 
					dataValue.name() + "," +
					this.classifier.measureNumLeaves() + "," +
					this.classifier.measureTreeSize() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
	}

}
