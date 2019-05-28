package experimento.weka;

import experimento.weka.base.DataSet;
import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import weka.classifiers.evaluation.Evaluation;
import weka.core.SelectedTag;

/**
 * Hello world!
 *
 */
public class AppBestMLP {
	
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
			
			//Classe para avaliação do classificador
			Evaluation eval;
			
			//======================================================================================
			//ARTIFICIAL NEURAL NETWORK
			//======================================================================================
			
			ArtificialNeuralNetwork mlp = new ArtificialNeuralNetwork();
			
			folds = 10;
			
			int[] seeds = {5,10,15,20,25};
			
			String mlpCSV = mlp.toCSVHeader();
			
			for(Integer s : seeds) {
				
				mlp.setSeed(s);
				
				System.out.println("EXECUTE MLP" + 
									" TRAINING TIME: " + 1000 + 
									" HIDDEN LAYERS : " + 90 + 
									" LAYERS RATE : " + 0.01 +
									" CLASSIFIER SEED:" + s +
									" DATA : " + dataValue);
						
				eval = mlp.evaluateClassifier(1000,90,0.01,folds,seed,dataValue,dataSet.getData());
				mlpCSV+=mlp.toCSV(eval, seed, folds, dataValue);
						
				mlp.toCSVFile(mlpCSV, "results/resultsBestMLP.csv");
				
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
