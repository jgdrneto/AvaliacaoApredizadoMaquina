package experimento.weka;

import experimento.weka.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;

/**
 * Hello world!
 *
 */
public class App {
	
    public static void main( String[] args ){

        try {
			DataSet dataSet = new DataSet("data.arff", DATAVALUE.NORMALIZED);
			
			KNearestNeighbors knn = new KNearestNeighbors();
			
			Evaluation eval = knn.evaluateClassifier(2,10,10,KNearestNeighbors.WEIGHT_NONE, dataSet.getData());
			
		    System.out.println(eval.toSummaryString("=== " + 10 + "-fold Cross-validation ===", false));
		 
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
