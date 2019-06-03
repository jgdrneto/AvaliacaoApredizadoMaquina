package experimento.weka.comite;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import experimento.weka.base.DataSet;
import experimento.weka.base.DataSet.DATAVALUE;
import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import experimento.weka.supervisionados.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
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
			int folds = 10;					//Quantidade de folds para o Cross-Validation
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
			//committees.add(new HomogeneousStackingCommittee());
			committees.add(new HeterogeneousStackingCommittee());
			
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
			Map<MyCommittee,BetterStackingData> bestsStackings = new HashMap<MyCommittee,BetterStackingData>();
			
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
								
								Evaluation e = execute(mc,listC,qClassifiers, folds, seed, data, datavalue, df,"results/comites/"+mc.getClass().getSimpleName()+".csv");
								
								if(mc instanceof HomogeneousStackingCommittee) {
									if(bestsStackings.isEmpty()) {
										bestsStackings.put(mc, new BetterStackingData(seed, listC, folds, datavalue, e.pctCorrect()));
									}else {
										if(e.pctCorrect()>bestsStackings.get(mc).getPrecision()) {
											bestsStackings.put(mc, new BetterStackingData(seed, listC, folds, datavalue, e.pctCorrect()));
										}
									}
								}
								
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
								
								Evaluation e = execute(mc,listC,qClassifiers, folds, seed, data, datavalue, df,"results/comites/"+mc.getClass().getSimpleName()+".csv");
								
								if(bestsStackings.isEmpty()) {
									bestsStackings.put(mc, new BetterStackingData(percentTrainingSet, seed, listC, folds, datavalue, l, e.pctCorrect()));
								}else {
									if(e.pctCorrect() > bestsStackings.get(mc).getPrecision()) {
										bestsStackings.put(mc, new BetterStackingData(percentTrainingSet, seed, listC, folds, datavalue, l, e.pctCorrect()));
									}
								}
							}				
						}
					}	
				}
			}
			//===================================END OBTAINING VALUES OF COMMITTEES==========================================
			
			//===================================INIT OBTAINING VALUES OF COMMITTEES CHANGE ATTRIBUTES SETS==========================================
			
			System.out.println("=======================================================");
			System.out.println("Obtains precisions from bests stackings committees");
			System.out.println("=======================================================");
			
			Map<String,Instances> datas = new HashMap<String,Instances>(2);
			datas.put("NATURAL", data);
			datas.put("ATTSUBSET:"+basePercent+"%_Seed:"+seeds[0], dataSet.getSubAttDataSet(seeds[0], basePercent));
			
			//Bests Stackings
			for(MyCommittee mc : bestsStackings.keySet()) {
				
				System.out.println("-------------------------------------"+mc.getClass().getSimpleName()+"-------------------------------------");
				
				for(String dataName : datas.keySet()) {
					
					//Obter os valores que dão a melhor precisão para o stacking
					BetterStackingData betterData = bestsStackings.get(mc);
					
					//Obtendo a lista de classificadores associados ao stacking
					List<MyClassifier> cs = betterData.getClassifier();
					
					System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + betterData.getPercentClassifiers() + " Number : " + cs.size() + " DATA : "+ datas.get(dataName) +" Seed : " + betterData.getSeed() +" Time : ");
					
					execute(mc,cs,cs.size(), folds, betterData.getSeed(), datas.get(dataName), betterData.getDatavalue(), df,"results/comites/Bests"+mc.getClass().getSimpleName()+"AttSubSet"+".csv");
				
				}
				System.out.println("-------------------------------------"+mc.getClass().getSimpleName()+"-------------------------------------");
				
			}
			
			//===================================END OBTAINING VALUES OF COMMITTEES CHANGE ATTRIBUTES SETS==========================================
			
			//===================================INIT CALCULATION OF PAIRWASE Q CLASSIFICATION==========================================
			
			System.out.println("=======================================================");
			System.out.println("Calcule Pairwise Q classification");
			System.out.println("=======================================================");
			
			
			Map<String,MyCommittee> comites = new HashMap<String,MyCommittee>();
			
			//Classificar classificadores diferentes
			//Sequencia: 0 - Homogeneo Normal
			//			 1 - Homogeneo SubAtt
			//			 2 - Heterogeneo Normal
			//			 3 - Heterogeneo SubAtt
			for(MyCommittee mc : bestsStackings.keySet()) {
				for(String dataName : datas.keySet()) {
					MyCommittee newMc = (MyCommittee) mc.copy();
					newMc.getClassifier().buildClassifier(datas.get(dataName));
					comites.put(mc.getcClass().getSimpleName()+"_"+dataName,newMc);
				}
			}
			
			int aCorreto = 0;
			int aIncorreto = 1;
			int bCorreto = 0;
			int bIncorreto = 1;
			
			Map<String,Map<String,Double>> valuesQ = new HashMap<String, Map<String, Double>>();
 			
			List<String> keys = new ArrayList<String>(comites.keySet());
			
			for(int c1 = 0;c1<keys.size()-1;c1++) {
				MyCommittee ca = comites.get(keys.get(c1));
				valuesQ.put(keys.get(c1), new HashMap<String, Double>());
				for(int c2=c1+1;c2<keys.size();c2++) {	
					int[][] matrix = {{0,0},{0,0}};
					
					MyCommittee cb = comites.get(keys.get(c2));
					
					for(Instance i : data) {
						
						int aclass = (int)ca.getClassifier().classifyInstance(i);
						int bclass = (int)cb.getClassifier().classifyInstance(i);
						int realclass = (int)i.classValue();
							
						if(bclass == realclass && bclass == realclass) {
							matrix[bCorreto][aCorreto]+=1;
						}else if (bclass == realclass && aclass != realclass) {
							matrix[bCorreto][aIncorreto]+=1;
						}else if(bclass != realclass && aclass == realclass) {
							matrix[bIncorreto][aCorreto]+=1;
						}else {
							matrix[bIncorreto][aIncorreto]+=1;
						}
					}
					
					double a = matrix[bCorreto][aCorreto];
					double b = matrix[bCorreto][aIncorreto];
					double c = matrix[bIncorreto][aCorreto];
					double d = matrix[bIncorreto][aIncorreto];
					
					double q = (a*d - b*c)/(a*d + b*c);
					
					valuesQ.get(keys.get(c1)).put(keys.get(c2), q);
				}
			}
			
			for(String c1 : valuesQ.keySet()) {
				for(String c2 : valuesQ.get(c1).keySet()) {
					System.out.println("Q["+c1+"]["+c2+"] = " + valuesQ.get(c1).get(c2));
				}
			}
			
			//===================================END CALCULATION OF PAIRWASE Q CLASSIFICATION==========================================
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
