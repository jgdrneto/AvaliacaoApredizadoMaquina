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
	
	public static Evaluation execute(MyCommittee mc,List<MyClassifier> listC,int qClassifiers,int folds, int seed, Instances data, DATAVALUE datavalue, DecimalFormat df, String filename) throws Exception {
		
		double start = ((double)System.currentTimeMillis())/1000; 
		
		Evaluation e = mc.evaluateClassifier(folds, listC, qClassifiers, seed, data,datavalue); 
		
		System.out.println( df.format( ((double)System.currentTimeMillis())/1000 - start) + "s");
		
		System.out.println("Quant Acertada : " + e.correct() + "("+df.format(e.pctCorrect())+"%)" + " Quant errada : " + e.incorrect() + "("+df.format(e.pctIncorrect())+"%)");
		
		mc.toSaveCSVFile(e,seed,folds,datavalue,filename);
		
		return e;
	}
	
	public static void main( String[] args ){
		
		try {
			//===================================INIT DEFINITIONS==========================================
			
			int percentTrainingSet = 30;	//Porcentagem da base de dados de treinamento para o bagging 
			int folds = 4;					//Quantidade de folds para o Cross-Validation
			int[] seeds = {3,5,7};			//Quantidade e valores de sementes usada para cada comite
			int basePercent = 50;			//Porcentagem da quantidade de atributos da base de dados a ser usada nos melhores comites

			//Definição se os dados vão ser normalizados ou não
			DATAVALUE datavalue = DATAVALUE.NOT_NORMALIZED;		
			
			//Lista com todos os melhores classificadores do melhor para o pior
			List<MyClassifier> classifiers = new ArrayList<MyClassifier>();	
			classifiers.add(new DecisionTree(false));
			classifiers.add(new KNearestNeighbors(1,KNearestNeighbors.WEIGHT_NONE));
			classifiers.add(new ArtificialNeuralNetwork(100,9,0.1));
			classifiers.add(new NaiveBayes());
			
			//Lista com todos os comites
			List<MyCommittee> committees = new ArrayList<MyCommittee>();
			//committees.add(new BaggingCommittee(percentTrainingSet));
			//committees.add(new BoostingCommittee());
			committees.add(new HomogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1)));
			committees.add(new HeterogeneousStackingCommittee(new ArtificialNeuralNetwork(100,9,0.1)));
			
			//Formatar os números para impressão com apenas 4 casas decimais
			DecimalFormat df = new DecimalFormat("0.0000");
			
			//Base de dados
			DataSet dataSet = new DataSet("data.arff");
			
			//Instâncias da base de dados
			Instances data = dataSet.getData();
			
			//Quantidade de classifadores utilizados no comite 
			int[] quantClassifiers = {10,15,20};
			
			//Porcentagem de cada classificador para o stacking heterogêneo respeitando ordem da lista classifiers   
			double[][] percentClassifier = {{50,50,0},{50,0,50},{0,50,50},{33,33,33}};
			
			//===================================END DEFINITIONS==========================================
			
			//===================================INIT OBTAINING VALUES OF COMMITTEES==========================================
			for(MyCommittee mc : committees) {
				for(Integer qClassifiers : quantClassifiers) {
					for(int seed : seeds) {
						if(!(mc instanceof HeterogeneousStackingCommittee)) {
							for(MyClassifier c : classifiers) {
								
								List<MyClassifier> listC = new ArrayList<MyClassifier>();
								
								if(mc instanceof HomogeneousStackingCommittee) {	
									for(int i=0;i<qClassifiers;i++) {
										listC.add(c.copy());
									}
								}else {
									listC.add(c.copy());
								}
								
								System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + c.getClass().getSimpleName() + " Number : " + qClassifiers + " Seed : " + seed + " Time : ");
								
								execute(mc,listC,qClassifiers, folds, seed, data, datavalue, df,"results/comites/"+mc.getClass().getSimpleName()+".csv");
							}
						}else {
							for(double[] l  : percentClassifier) {
								
								List<Double> lList = new ArrayList<Double>();
								
								for(double cl : l) {
									lList.add(cl);
								}
								
								int[] c = new int[l.length];
								
								System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + lList.toString() + " Number : " + qClassifiers + " Seed : " + seed +" Time : ");
								
								int count = 0;
								for(int i=0;i<l.length;i++) {
									c[i] = (int)(qClassifiers * (l[i]/100));
									count+=c[i];
								}
													
								int dif = qClassifiers - count;
								
								for(int i=0;i<c.length;i++) {
									if(dif>0 && c[i]!=0) {
										c[i]++;
										dif--;
									}
								}
								
								List<MyClassifier> listC = new ArrayList<MyClassifier>();
								
								//Criando os classificadores de acordo com a quantidade e porcentagem
								for(int i=0;i<c.length;i++) {
									for(int j=0;j<c[i];j++) {
										listC.add(classifiers.get(i).copy());
									}
								}
								
								execute(mc,listC,qClassifiers, folds, seed, data, datavalue, df,"results/comites/"+mc.getClass().getSimpleName()+".csv");
							}				
						}
					}	
				}
			}
			
			//===================================END OBTAINING VALUES OF COMMITTEES==========================================
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
