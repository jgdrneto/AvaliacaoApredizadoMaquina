package experimento.weka;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;

public class KNearestNeighbors extends IBk{
	
	public static SelectedTag WEIGHT_INVERSE = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
	public static SelectedTag WEIGHT_NONE = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
	public static SelectedTag WEIGHT_SIMILARITY = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);

	private static final long serialVersionUID = 1L;
	
	public KNearestNeighbors() throws Exception {
		super();
	}
	
	public Evaluation evaluateClassifier(int k,int folds, int seed, SelectedTag weight, Instances data) throws Exception{
		
		this.setKNN(k);
		
		this.setDistanceWeighting(weight);
		
		Random rand = new Random(seed);
	    Instances randData = new Instances(data);
	    randData.randomize(rand);
		
		if (randData.classAttribute().isNominal()) {
		      randData.stratify(folds);
		} 
		
		Evaluation eval = new Evaluation(randData);
		
		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n, rand);
			Instances test = randData.testCV(folds, n);

			Classifier clsCopy = AbstractClassifier.makeCopy(this);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);
		}
		
		return eval;
	}
	
}
