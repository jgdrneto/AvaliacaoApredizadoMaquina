package experimento.weka;

import experimento.weka.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.core.SelectedTag;

/**
 * Hello world!
 *
 */
public class App {
	
    public static void main( String[] args ){
    	
        try {
        	
        	//DATASET
			DataSet dataSet = new DataSet("data.arff");
			
			//INT SEED
			int seed = 10;
			
			//DATAVALUE
			DATAVALUE dataValue = DATAVALUE.NOT_NORMALIZED;
			
			//FOLDS
			int folds = 10;
			
			//=====================================================================================
			//KNN EXPERIMENTS
			//=====================================================================================
			KNearestNeighbors knn = new KNearestNeighbors(); 
			
			String knncsv = knn.toCSVHeader();
			
			for(int j=0;j<2;j++) {
				
				switch(j) {
					case 0:
						dataValue = DATAVALUE.NOT_NORMALIZED;
					break;
					default:
						dataValue = DATAVALUE.NORMALIZED;
				}
				
				SelectedTag tag;
					
				for(int k=0;k<3;k++) {
						
					switch(k) {
						case 0:
							tag = KNearestNeighbors.WEIGHT_NONE;
						break;
						case 1:
							tag = KNearestNeighbors.WEIGHT_INVERSE;
						break;
						default:
							tag = KNearestNeighbors.WEIGHT_SIMILARITY;
					}
					
					for(int i=2; i<12;i++) {
						
						System.out.println("K : " + i + " WEIGHT : " + tag.getSelectedTag().getReadable() + " DATA : " + dataValue);
						
						Evaluation eval = knn.evaluateClassifier(i,tag,folds,seed,dataValue, dataSet.getData());
						knncsv+=knn.toCSV(eval, seed, folds, dataValue);
					}	
				}
			}
			
			knn.toCSVFile(knncsv, "results/knn.csv");

			dataValue = DATAVALUE.NOT_NORMALIZED;
			
			//======================================================================================
			//DECISION TREE EXPERIMENTS
			//=====================================================================================
			DecisionTree dt = new DecisionTree();
			
			String dtCSV = dt.toCSVHeader();
			
			boolean poda;
			
			for(int i=0;i<2;i++) {
				switch(i) {
					case 0:
						poda = false;
					break;
					default:
						poda = true;
				}
				
				System.out.println("UNPRUNED : " + poda + " DATA : " + dataValue);
				
				Evaluation eval = dt.evaluateClassifier(poda,folds,seed,dataValue, dataSet.getData());
				dtCSV+=dt.toCSV(eval, seed, folds, dataValue);
			}
				
			dt.toCSVFile(dtCSV, "results/decisionTree.csv");
			
			
			/*
			NaiveBayes nb = new NaiveBayes();
			
			Evaluation eval = nb.evaluateClassifier(10,10,DATAVALUE.NORMALIZED, dataSet.getData());
			
			//Evaluation eval = knn.evaluateClassifier(2,KNearestNeighbors.WEIGHT_NONE,10,10,DATAVALUE.NORMALIZED, dataSet.getData());
			
		    System.out.println(eval.toSummaryString("=== " + 10 + "-fold Cross-validation ===", false));
		    
		    System.out.println("Correct classified: " + eval.pctCorrect());
		    */
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
