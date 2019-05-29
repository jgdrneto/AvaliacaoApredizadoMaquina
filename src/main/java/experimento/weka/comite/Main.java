package experimento.weka.comite;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import experimento.weka.base.DataSet;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import experimento.weka.supervisionados.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class Main {
	
	public static void main( String[] args ){
		
		try {

			int basePercent = 10;
			int folds = 10;
			int seed = 3;
			
			List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
			classifiers.add(new NaiveBayes());
			//classifiers.add(new DecisionTree(false));
			//classifiers.add(new KNearestNeighbors(1,KNearestNeighbors.WEIGHT_NONE));
			//classifiers.add(new ArtificialNeuralNetwork(100,9,0.1));
			
			List<MyCommittee> committees = new ArrayList<MyCommittee>();
			committees.add(new BaggingCommittee());

			DecimalFormat df = new DecimalFormat("0.000");
			
			DataSet dataSet = new DataSet("data.arff");
		
			Instances data = dataSet.getData();
			
			int[] quantClassifiers = {10};
			
			for(MyCommittee mc : committees) {
				for(Integer qClassifiers : quantClassifiers) {
					for(MyClassifier c : classifiers) {
						
						System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + c.getClass().getSimpleName() + " Number : " + qClassifiers + " Accuracy : ");
						
						
						
						//Evaluation e = mc.evaluateClassifier(folds, c, qClassifiers, seed, data);
						
						//System.out.println(df.format(e.pctCorrect()));
						
						/*
						eval.crossValidateModel(mc.getClassifier(), data, folds, new Random(seed));
						System.out.println(df.format(eval.pctCorrect()));
						System.out.println(mc.getClassifier());
						System.out.println(eval.toSummaryString(true));
						System.out.println(eval.toMatrixString());
						System.out.println(eval.toClassDetailsString());
						*/
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
