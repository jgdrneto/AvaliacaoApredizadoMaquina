package experimento.weka.base;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public abstract class MyClassifier {
	
	public abstract String toCSVHeader();
	public abstract String toCSV(Evaluation e, int seed, int folds,DATAVALUE dataValue);
	public abstract Classifier getClassifier();
	
	protected Evaluation evaluateClassifier(int folds,Classifier classifier, int seed,DATAVALUE datavalue, Instances data) throws Exception{
		
		if(datavalue==DATAVALUE.NORMALIZED) {
			Filter filterNorm = new Normalize();
	        filterNorm.setInputFormat(data);
	        data = Filter.useFilter(data, filterNorm);
		}
		
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

			Classifier clsCopy = AbstractClassifier.makeCopy(classifier);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);
		}
		
		return eval;
	}
	
	public void toCSVFile(String csv,String filename) {
		
		File file = new File(filename);
		
		if(file.getParentFile()!=null) {
			file.getParentFile().mkdirs();
		}
		
		try {
			Writer writer = new FileWriter(file);
			
			writer.write(csv);
					
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
}
