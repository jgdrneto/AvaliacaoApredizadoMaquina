package experimento.weka.comite;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import experimento.weka.supervisionados.ArtificialNeuralNetwork;
import experimento.weka.supervisionados.DecisionTree;
import experimento.weka.supervisionados.KNearestNeighbors;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class BaggingCommittee extends MyCommittee{
	
	public int PERCENTAGE = 10;
	
	public double executeCommittee(Class<? extends ArtificialNeuralNetwork> cClass,int seed,int quantClassifier,int ciclos,int neuroniosEscondidos,double taxaAprendizado,Instances data, int folds) throws Exception {
		
		List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
		
		for(int i=0;i<quantClassifier;i++) {
			classifiers.add(new ArtificialNeuralNetwork(ciclos,neuroniosEscondidos,taxaAprendizado));
		}
		
		return auxExecuteCommittee(classifiers, cClass, seed, folds, data);
		
	}
	
	public double executeCommittee(Class<? extends DecisionTree> cClass, int seed,int quantClassifier,boolean unpruned, Instances data, int folds) throws Exception {
		
		List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
		
		for(int i=0;i<quantClassifier;i++) {
			classifiers.add(new DecisionTree(unpruned));
		}
		
		return auxExecuteCommittee(classifiers, cClass, seed, folds, data);
	}
	
	public double executeCommittee(Class<? extends KNearestNeighbors> cClass, int seed,int quantClassifier,int k,SelectedTag weight,Instances data, int folds) throws Exception {
		
		List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
		
		for(int i=0;i<quantClassifier;i++) {
			classifiers.add(new KNearestNeighbors(k,weight));
		}
		
		return auxExecuteCommittee(classifiers, cClass, seed, folds, data);
		
	}
	
	public double executeCommittee(Class<? extends experimento.weka.supervisionados.NaiveBayes> cClass, int seed,int quantClassifier, Instances data, int folds) throws Exception {
		
		List<MyClassifier> classifiers = new ArrayList<MyClassifier>();
		
		for(int i=0;i<quantClassifier;i++) {
			classifiers.add(new experimento.weka.supervisionados.NaiveBayes());
		}
		
		return auxExecuteCommittee(classifiers, cClass, seed, folds, data);
	}
	
	public double auxExecuteCommittee(List<MyClassifier> classifiers, Class<? extends MyClassifier> cClass, int seed, int folds, Instances data) throws Exception {
		
		DecimalFormat df = new DecimalFormat("0.0000"); 
		
		Random rand = new Random(seed);
		
		double percentGlobal = 0.0;
		
		System.out.println("==============================================================");
		
		System.out.println("Bagging Committee " + cClass.getSimpleName());
		
		System.out.println("==============================================================");
		
		for (int n = 0; n < folds; n++) {
			
			double startTime = ((double)System.currentTimeMillis())/1000;
			
			Map<MyClassifier,Instances> comite = new HashMap<MyClassifier,Instances>(classifiers.size());
			
			System.out.print("Fold "+ n + " : ");
			
			Instances train = data.trainCV(folds, n, rand);
			Instances test = data.testCV(folds, n);
			
			List<Instances> datas = this.obtainsDataSets(train,PERCENTAGE,classifiers.size(), seed);
			
			//Construir classificadores
			for(int i=0;i<classifiers.size();i++) {
				
				classifiers.get(i).buildClassifier(datas.get(i));
				comite.put(classifiers.get(i), datas.get(i));
			}
			
			double resultFold =executeCommitteeTest(comite,seed,test);
			
			System.out.print(resultFold);
			
			percentGlobal+=(resultFold/folds); 
			
			System.out.println(" Time: " + df.format((((double)System.currentTimeMillis()/1000) - startTime)) + "s" );
			
			System.out.println("-----------------------------------------------------------");
		}
		
		return percentGlobal;
		
	}
	
	private List<Instances> obtainsDataSets(Instances data,int percent, int quantClass, int seed) throws Exception {
		
		List<Instances> result = new ArrayList<Instances>();
		
		Random rand = new Random(seed);
		
		for(int i=0;i<quantClass;i++) {
			Resample resampleFilter = new Resample();
			resampleFilter.setInputFormat(data);
			resampleFilter.setSampleSizePercent(percent);
			resampleFilter.setRandomSeed(rand.nextInt());
			result.add(Filter.useFilter(data, resampleFilter));
		}
		/*
		for(Instances i : result) {
			System.out.println("indice 1 : " + i.get(0).toString() + "Tamanho:" + i.size());
		}
		*/
		return result;
	}

	private double executeCommitteeTest(Map<MyClassifier,Instances> comite, int seed, Instances data) throws Exception {
		
		double pctCorrect = 0;
		
		List<Double> classifications = new ArrayList<Double>(); 
		
		//Alocar valores das classes
		for(int i=0;i<data.numClasses();i++) {
			classifications.add(0.0);
		}
		
		//Realizar votação de maioria simpĺes
		for(Instance i : data) {
			for(MyClassifier c : comite.keySet()) {
				int classInst = (int)c.classifyInstance(i);
				classifications.set(classInst, classifications.get(classInst)+1);
			}
			
			double maioriaVotos = classifications.indexOf(Collections.max(classifications));
			
			//System.out.println("Maioria votos: " + Collections.max(classifications) + " Classe do classificador : " + maioriaVotos + " Classe real: " + i.classValue());
			
			if(maioriaVotos == i.classValue()) {
				pctCorrect+=(1.0/data.size());
			}
			
			//Limpar valores
			for(int v=0;v<data.numClasses();v++) {
				classifications.set(v,0.0);
			}
		}
		
		return pctCorrect;
		
	}

}
