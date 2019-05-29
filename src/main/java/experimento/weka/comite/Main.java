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
	
	public static void main( String[] args ){
		
		try {

			int basePercent = 100;
			int folds = 10;
			int seed = 3;
			DATAVALUE datavalue = DATAVALUE.NOT_NORMALIZED;
			
			List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
			classifiers.add(new DecisionTree(false));
			classifiers.add(new KNearestNeighbors(1,KNearestNeighbors.WEIGHT_NONE));
			classifiers.add(new ArtificialNeuralNetwork(100,9,0.1));
			classifiers.add(new NaiveBayes());
			
			List<MyCommittee> committees = new ArrayList<MyCommittee>();
			//committees.add(new BaggingCommittee());
			//committees.add(new BoostingCommittee());
			committees.add(new HomogeneousStackingCommittee());
			committees.add(new HeterogeneousStackingCommittee());
			
			DecimalFormat df = new DecimalFormat("0.0000");
			
			DataSet dataSet = new DataSet("data.arff");
		
			Instances data = dataSet.getSubDataSet(seed, basePercent);
			
			int[] quantClassifiers = {10,15,20};
			
			for(MyCommittee mc : committees) {
				for(Integer qClassifiers : quantClassifiers) {
					
					if(!(mc instanceof HeterogeneousStackingCommittee)) {
						for(MyClassifier c : classifiers) {
							
							List<MyClassifier> listC = new ArrayList<MyClassifier>();
							
							if(mc instanceof HomogeneousStackingCommittee) {	
								for(int i=0;i<qClassifiers;i++) {
									listC.add(c.copy());
								}
							}else {
								listC.add(c);
							}
							
							double start = ((double)System.currentTimeMillis())/1000; 
							
							System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + c.getClass().getSimpleName() + " Number : " + qClassifiers + " Time : ");
							
							Evaluation e = mc.evaluateClassifier(folds, listC, qClassifiers, seed, data,datavalue); 
							
							System.out.println( df.format( ((double)System.currentTimeMillis())/1000 - start) + "s");
							
							System.out.println("Quant Acertada : " + e.correct() + "("+df.format(e.pctCorrect())+"%)" + " Quant errada : " + e.incorrect() + "("+df.format(e.pctIncorrect())+"%)");
							
							mc.toSaveCSVFile(e,seed,folds,DATAVALUE.NOT_NORMALIZED,"results/comites/"+mc.getClass().getSimpleName()+".csv");
						}
					}else {
						double[][] valores = {{50,50,0},{50,0,50},{0,50,50},{33,33,33}};
						
						for(double[] l  : valores) {
							
							int[] c = new int[l.length];
							
							System.out.print("Committee : " + mc.getClass().getSimpleName() +" Classifier : " + l.toString() + " Number : " + qClassifiers + " Time : ");
							
							int count =0;
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
							
							double start = ((double)System.currentTimeMillis())/1000; 
							
							Evaluation e = mc.evaluateClassifier(folds, listC, qClassifiers, seed, data,datavalue); 
							
							System.out.println( df.format( ((double)System.currentTimeMillis())/1000 - start) + "s");
							
							System.out.println("Quant Acertada : " + e.correct() + "("+df.format(e.pctCorrect())+"%)" + " Quant errada : " + e.incorrect() + "("+df.format(e.pctIncorrect())+"%)");
							
							mc.toSaveCSVFile(e,seed,folds,DATAVALUE.NOT_NORMALIZED,"results/comites/"+mc.getClass().getSimpleName()+".csv");
							
						}
						
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
}
