package experimento.weka.comite;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import experimento.weka.base.DataSet;
import experimento.weka.base.DataSet.DATAVALUE;
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
			DATAVALUE datavalue = DATAVALUE.NOT_NORMALIZED;
			
			List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
			classifiers.add(new NaiveBayes());
			classifiers.add(new DecisionTree(false));
			classifiers.add(new KNearestNeighbors(1,KNearestNeighbors.WEIGHT_NONE));
			classifiers.add(new ArtificialNeuralNetwork(100,9,0.1));
			
			List<MyCommittee> committees = new ArrayList<MyCommittee>();
			committees.add(new BaggingCommittee());
			committees.add(new BoostingCommittee());
			
			DecimalFormat df = new DecimalFormat("0.0000");
			
			DataSet dataSet = new DataSet("data.arff");
		
			//Instances data = dataSet.getSubDataSet(seed,basePercent);
			Instances data = dataSet.getData();
			
			int[] quantClassifiers = {10,15,20};
			
			for(MyCommittee mc : committees) {
				for(Integer qClassifiers : quantClassifiers) {
					for(MyClassifier c : classifiers) {
						
						double start = ((double)System.currentTimeMillis())/1000; 
						
						System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + c.getClass().getSimpleName() + " Number : " + qClassifiers + " Time : ");
						
						Evaluation e = mc.evaluateClassifier(folds, c, qClassifiers, seed, data,datavalue); 
						
						System.out.println( df.format( ((double)System.currentTimeMillis())/1000 - start) + "s");
						
						//System.out.println("Quant Acertada : " + e.correct() + " Quant errada : " + e.incorrect());
						
						mc.toSaveCSVFile(e,seed,folds,DATAVALUE.NOT_NORMALIZED,"results/comites/"+mc.getClass().getSimpleName()+".csv");
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
