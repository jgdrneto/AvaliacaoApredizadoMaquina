package experimento.weka.naosupervisionados;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class KNearestNeighbors extends MyClassifier{
	
	public static SelectedTag WEIGHT_INVERSE = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
	public static SelectedTag WEIGHT_NONE = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
	public static SelectedTag WEIGHT_SIMILARITY = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);
	
	public IBk classifier;
	
	public KNearestNeighbors() throws Exception {
		this.classifier = new IBk();
	}
	
	public void buildClassifier(int k,SelectedTag weight,Instances data) throws Exception {
		
		this.classifier.setKNN(k);
		this.classifier.setDistanceWeighting(weight);
		
		super.buildClassifier(classifier,data);
		
	}
	
	@Override
	public double classifyInstance(Instance i) throws Exception {
		return this.classifier.classifyInstance(i);
	}
	
	public Evaluation evaluateClassifier(int k,SelectedTag weight,int folds, int seed, DATAVALUE datavalue, Instances data) throws Exception{
		
		this.classifier.setKNN(k);
		
		this.classifier.setDistanceWeighting(weight);
		
		return this.evaluateClassifier(folds, this.classifier, seed, datavalue, data);
	}
	
	@Override
	public String toCSVHeader() {
		
		return "K_VALUE,WEIGHT,FOLDS,SEED,DATA,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED\n\n";
		
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds,DATAVALUE dataValue) {
		
		String csv =this.classifier.getKNN() + "," + 
					this.classifier.getDistanceWeighting().getSelectedTag().getReadable()+ "," +
					folds + "," +
					seed + "," + 
					dataValue.name() + "," +
					e.pctCorrect() + "," +
					e.pctIncorrect() + "\n";

		return csv;
	}
	
}
