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
import experimento.weka.base.MyCommittee;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Hello world!
 *
 */
public class AppQmetric {
	
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
			
			//===================================INIT CALCULATION OF PAIRWASE Q CLASSIFICATION==========================================
			
			Map<String,MyCommittee> comites = new HashMap<String,MyCommittee>();
			
			System.out.println("CREATE HomogeneousStackingCommittee WITH NORMAL DATABASE");
			comites.put("HOMO_NORMAL",new HomogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1),homoClassifiersCommittee,size,seeds[1],data));
			System.out.println("CREATE HomogeneousStackingCommittee WITH SUBSETATT DATABASE");
			comites.put("HOMO_ATTSUBSET",new HomogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1),homoClassifiersCommittee,size,seeds[1],datasubAtt));
			System.out.println("CREATE HeterogeneousStackingCommittee WITH NORMAL DATABASE");
			comites.put("HETERO_NORMAL",new HeterogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1),heteroClassifiersCommittee,size,seeds[1],data));
			System.out.println("CREATE HeterogeneousStackingCommittee WITH SUBSETATT DATABASE");
			comites.put("HETERO_ATTSUBSET",new HeterogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1),heteroClassifiersCommittee,size,seeds[1],datasubAtt));
			
			int aCorreto = 0;
			int aIncorreto = 1;
			int bCorreto = 0;
			int bIncorreto = 1;
			
			Map<String, Double> results = new HashMap<String, Double>();
			
			for(String key : comites.keySet()) {
				
				MyCommittee  mc = comites.get(key);
				
				Classifier[] cs = mc.getClassifiers();
				
				List<Double> valuesQ = new ArrayList<Double>();
	 			
				for(int c1 = 0;c1<cs.length-1;c1++) {
					
					Classifier ca = cs[c1]; 
					
					for(int c2=c1+1;c2<cs.length;c2++) {	
						System.out.println("Calcule Q to "+ key + ": ["+ c1+ "]" + "["+ c2+ "]");
						int[][] matrix = {{0,0},{0,0}};
						
						Classifier cb = cs[c2]; 
						
						for(int i = 0; i<data.size();i++) {
							int aclass;
							int bclass;
							
							if(key.contains("ATTSUBSET")) {
								aclass = (int)ca.classifyInstance(datasubAtt.get(i));
								bclass = (int)cb.classifyInstance(datasubAtt.get(i));
							}else {
								aclass = (int)ca.classifyInstance(data.get(i));
								bclass = (int)cb.classifyInstance(data.get(i));
							}
							
							int realclass = (int)data.get(i).classValue();
							
							if(bclass == realclass && aclass == realclass) {
								matrix[bCorreto][aCorreto]+=1;
							}
							if (bclass == realclass && aclass != realclass) {
								matrix[bCorreto][aIncorreto]+=1;
							}
							if(bclass != realclass && aclass == realclass) {
								matrix[bIncorreto][aCorreto]+=1;
							}
							if(bclass != realclass && aclass != realclass){
								matrix[bIncorreto][aIncorreto]+=1;
							}
						}
						
						double a = matrix[bCorreto][aCorreto];
						double b = matrix[bCorreto][aIncorreto];
						double c = matrix[bIncorreto][aCorreto];
						double d = matrix[bIncorreto][aIncorreto];
						
						System.out.println("Matrix");
						System.out.println("[" + a + "]" + "[" + b + "]");
						System.out.println("[" + c + "]" + "[" + d + "]");
						
						double q;
						
						if((a*d + b*c)==0) {
							q = 1;
						}else {
							q = (a*d - b*c)/(a*d + b*c);
						}
						
						valuesQ.add(q);
					}
				}
				
				double media=0;
				
				for(Double d : valuesQ) {
					media+=d/valuesQ.size();
				}
				
				results.put(key, media);
				
			}
			
			for(String key : results.keySet()) {
				System.out.println("KEY: " + key + " Q valor: " + results.get(key));
			}
			
    	} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
