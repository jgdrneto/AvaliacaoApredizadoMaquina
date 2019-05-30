package experimento.weka.comite;

import java.util.List;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.DataSet.DATAVALUE;

public class BetterStackingData {
	
	int percentTrainingSet;
	int seed;
	List<MyClassifier> classifier;
	int folds;
	DATAVALUE datavalue;
	double[] percentClassifiers;
	double precision;
	
	public BetterStackingData(int seed,List<MyClassifier> classifier, int folds,
			DATAVALUE datavalue, double precision) {
		this.seed = seed;
		this.classifier = classifier;
		this.folds = folds;
		this.datavalue = datavalue;
		this.precision = precision;
	}

	public BetterStackingData(int percentTrainingSet, int seed, List<MyClassifier> classifier, int folds,
			DATAVALUE datavalue, double[] percentClassifiers, double precision) {
		super();
		this.percentTrainingSet = percentTrainingSet;
		this.seed = seed;
		this.classifier = classifier;
		this.folds = folds;
		this.datavalue = datavalue;
		this.percentClassifiers = percentClassifiers;
		this.precision = precision;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}

	public int getPercentTrainingSet() {
		return percentTrainingSet;
	}

	public int getSeed() {
		return seed;
	}

	public List<MyClassifier> getClassifier() {
		return classifier;
	}

	public int getFolds() {
		return folds;
	}

	public DATAVALUE getDatavalue() {
		return datavalue;
	}

	public double[] getPercentClassifiers() {
		return percentClassifiers;
	}		

}
