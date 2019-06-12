package experimento.weka;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import experimento.weka.base.DataSet;
import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.comite.HeterogeneousStackingCommittee;
import experimento.weka.comite.HomogeneousStackingCommittee;
import experimento.weka.comite.Main;
import experimento.weka.base.MyCommittee;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Hello world!
 *
 */
public class AppBestStackings {
	
    public static void main( String[] args ){
    	
    	try {
    		//Tamanho das Stackings
    		int size = 10;
    		
    		//Quantidade de Folds
    		int folds = 10;
    		
    		//Percentual do subconjunto de atributos
    		int basePercent = 50;
    		
    		//Quantidade e valores de sementes usada para cada comite
    		int[] seeds = {3,5,7};
    		
    		//Formatar os números para impressão com apenas 4 casas decimais
			DecimalFormat df = new DecimalFormat("0.0000");
    		
    		//Base de dados
    		DataSet dataSet = new DataSet("data.arff");
    		
    		//Definição se os dados vão ser normalizados ou não
			DATAVALUE datavalue = DATAVALUE.NOT_NORMALIZED;		
    		
    		//Instâncias da base de dados
			Instances data = dataSet.getData();
			
    		//Instancias com 50% dos atributos
			Instances datasubAtt = dataSet.getSubAttDataSet(seeds[0], basePercent);

			List<MyClassifier> homoClassifiersCommittee = new ArrayList<MyClassifier>(size);	
			for(int i=0;i<size;i++) {
				homoClassifiersCommittee.add(new DecisionTree(false));
			}
			
			List<MyClassifier> heteroClassifiersCommittee = new ArrayList<MyClassifier>(size);	
			
			for(int i=0;i<size;i++) {
				
				if(i<size/2) {
					heteroClassifiersCommittee.add(new DecisionTree(false));
				}else {
					heteroClassifiersCommittee.add(new KNearestNeighbors(1,KNearestNeighbors.WEIGHT_NONE));
				}
			}
			
			//Question6
			List<MyCommittee> q6committees = new ArrayList<MyCommittee>();
			q6committees.add(new HomogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1)));
			q6committees.add(new HeterogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1)));
				
			//Bests Stackings
			for(MyCommittee mc : q6committees) {
				String classifierValues;
				List<MyClassifier> cs;
				
				if(mc instanceof HomogeneousStackingCommittee) {
					classifierValues = "100% Decision Tree";
					cs = homoClassifiersCommittee;
				}else {
					classifierValues = "50% Decision Tree e 50% KNN";
					cs = heteroClassifiersCommittee;
				}
				
				System.out.println("-------------------------------------"+mc.getClass().getSimpleName()+"-------------------------------------");
				
				System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + classifierValues + " Number : " + size + " DATA : "+ "Aleatório " + basePercent + "% dos atributos" +" Seed : " + seeds[1] +" Time : ");
				
				Main.execute(mc,cs,cs.size(),folds, seeds[1], datasubAtt, datavalue, df,"results/comites/BestsStackingsAttSubSet"+".csv");

				System.out.println("-------------------------------------"+mc.getClass().getSimpleName()+"-------------------------------------");
				
			}
			
			//===================================END OBTAINING VALUES OF COMMITTEES CHANGE ATTRIBUTES SETS==========================================
    	} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
