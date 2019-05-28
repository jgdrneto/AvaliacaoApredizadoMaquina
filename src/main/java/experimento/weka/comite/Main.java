package experimento.weka.comite;

import experimento.weka.base.DataSet;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import experimento.weka.supervisionados.NaiveBayes;

public class Main {
	
	public static void main( String[] args ){
		
		try {
			
			DataSet dataSet = new DataSet("data.arff");
			
			int[] quantClassifiers = {10,15,20};
			
			BaggingCommittee comite = new BaggingCommittee();
			
			for(Integer qClassifiers : quantClassifiers) {
				System.out.println("NB Commite - " + "Number of Classifier : " + qClassifiers +" Accuracy : " + comite.executeCommittee(NaiveBayes.class, 3, qClassifiers, dataSet.getData(), 10));
				System.out.println("DT Commite - " + "Number of Classifier : " + qClassifiers +" Accuracy : " + comite.executeCommittee(DecisionTree.class, 3, qClassifiers, false, dataSet.getData(), 10));
				System.out.println("KNN Commite - " + "Number of Classifier : " + qClassifiers +" Accuracy : "  + comite.executeCommittee(KNearestNeighbors.class, 3, qClassifiers,1,KNearestNeighbors.WEIGHT_NONE, dataSet.getData(), 10));
				System.out.println("MLP Commite - " + "Number of Classifier : " + qClassifiers +" Accuracy : "  + comite.executeCommittee(ArtificialNeuralNetwork.class, 3, qClassifiers,100,40,0.1,dataSet.getData(), 10));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
