package experimento.weka.supervisionados;

import java.util.ArrayList;
import java.util.List;

import experimento.weka.base.DataSet;
import experimento.weka.base.MyClusterer;
import weka.core.Instances;


/**
 * Hello world!
 *
 */
public class Main {
	
    public static void main( String[] args ){
    	
        try {
			//VALORES DE SEMENTES
        	int[] seeds = {3,12,17,21,30}; 
        	int maxK = 20;
        	
        	List<MyClusterer> clusterer= new ArrayList<MyClusterer>();
        	//clusterer.add(new Kmeans());
        	clusterer.add(new EM());
        	//clusterer.add(new HA());
        	
        	//MUDAR PARA O ARQUIVO BASE DA SUA BASE DE DADOS
        	DataSet dataSet = new DataSet("data.arff");
			
        	Instances data = dataSet.getData();
			Instances dataNotClass = dataSet.getDataNotClassifier();
			
			for(MyClusterer c : clusterer) {
				for(int k =2;k<=maxK;k++) {
					if(!(c instanceof HA)) {
						for(int seed : seeds) {
							System.out.println ("Clusterer" + " : " + c.getClass().getSimpleName() + " K" + " : " + k + " seed" + " : " + seed);
							
							c.executeClustering(k, seed, data, dataNotClass);
							
							c.toSaveCSVFile(seed,"results/naosupervisionados/"+c.getClass().getSimpleName()+".csv");
						}	
					}else {
						System.out.println ("Clusterer" + " : " + c.getClass().getSimpleName() + " K" + " : " + k);
						c.executeClustering(k,0, data, dataNotClass);
						c.toSaveCSVFile(0,"results/naosupervisionados/"+c.getClass().getSimpleName()+".csv");
					}
				}
			}
			
        } catch (Exception e) {
			e.printStackTrace();
		}
		
    }    
}
