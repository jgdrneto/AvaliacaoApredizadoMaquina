package experimento.weka;

import experimento.weka.base.DataSet;
import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.naosupervisionados.ArtificialNeuralNetwork;
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
			
			//Classe para avaliação do classificador
			Evaluation eval;
			
			
			//=====================================================================================
			//KNN EXPERIMENTS
			//=====================================================================================
			/*
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
					
					for(int i=1; i<11;i++) {
						
						System.out.println("EXECUTE KNN " + "K : " + i + " WEIGHT : " + tag.getSelectedTag().getReadable() + " DATA : " + dataValue);
						
						eval = knn.evaluateClassifier(i,tag,folds,seed,dataValue, dataSet.getData());
						knncsv+=knn.toCSV(eval, seed, folds, dataValue);
					}	
				}
			}
			
			knn.toCSVFile(knncsv, "results/knn.csv");

			dataValue = DATAVALUE.NOT_NORMALIZED;
			*/
			/*
			//======================================================================================
			//DECISION TREE EXPERIMENTS
			//======================================================================================
			DecisionTree dt = new DecisionTree();
			
			String dtCSV = dt.toCSVHeader();
			
			boolean unpruned;
			
			for(int i=0;i<2;i++) {
				switch(i) {
					case 0:
						unpruned = false;
					break;
					default:
						unpruned = true;
				}
				
				System.out.println("EXECUTE DECISION TREE" + " UNPRUNED : " + unpruned + " DATA : " + dataValue);
				
		 		eval = dt.evaluateClassifier(unpruned,folds,seed,dataValue, dataSet.getData());
				dtCSV+=dt.toCSV(eval, seed, folds, dataValue);
			}
				
			dt.toCSVFile(dtCSV, "results/decisionTree.csv");
			
			//======================================================================================
			//NAIVE BAYES
			//======================================================================================
			
			NaiveBayes nb = new NaiveBayes();
			
			String nbCSV = nb.toCSVHeader();
			
			System.out.println("EXECUTE Naive Bayes" + " DATA : " + dataValue);
			
			eval = nb.evaluateClassifier(folds,seed,dataValue,dataSet.getData());
			
			nbCSV+=nb.toCSV(eval, seed, folds, dataValue);
			
			nb.toCSVFile(nbCSV, "results/naiveBayes.csv");
			*/
			//======================================================================================
			//ARTIFICIAL NEURAL NETWORK
			//======================================================================================
			
			ArtificialNeuralNetwork mlp = new ArtificialNeuralNetwork();
			
			folds = 2;
			
			String mlpCSV = mlp.toCSVHeader();
			
			for(int i=1;i<3;i++) {
				
				int ciclos;
				
				switch(i) {
					case 0:
						ciclos = 100;
					break;
					case 1:
						ciclos = 1000;
					break;
					default:
						ciclos = 5000;
				}
				
				for(int j=0;j<3;j++) {
					
					int neuroniosEscondidos=90+10*j;
					
					for(int k =0;k<3;k++) {
						
						double taxaAprendizado;
						
						switch(k) {
							case 0:
								taxaAprendizado = 0.1;
							break;
							case 1:
								taxaAprendizado = 0.01;
							break;
							default:
								taxaAprendizado = 0.001;
						}
						
						System.out.println("EXECUTE MLP" + 
											" TRAINING TIME: " + ciclos + 
											" HIDDEN LAYERS : " + neuroniosEscondidos + 
											" LAYERS RATE : " + taxaAprendizado +
											" DATA : " + dataValue);
						
						eval = mlp.evaluateClassifier(ciclos,neuroniosEscondidos,taxaAprendizado,folds,seed,dataValue,dataSet.getData());
						mlpCSV+=mlp.toCSV(eval, seed, folds, dataValue);
						
						mlp.toCSVFile(mlpCSV, "results/resultsMLP.csv");
					}
				}
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
