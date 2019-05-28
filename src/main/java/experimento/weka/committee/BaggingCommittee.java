package experimento.weka.committee;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import experimento.weka.base.MyClassifier;
import experimento.weka.base.MyCommittee;
import experimento.weka.naosupervisionados.ArtificialNeuralNetwork;
import experimento.weka.naosupervisionados.DecisionTree;
import experimento.weka.naosupervisionados.KNearestNeighbors;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public abstract class BaggingCommittee extends MyCommittee{
	
	public double executeCommittee(Class<? extends ArtificialNeuralNetwork> cClass,int seed,int quantClassifier,int ciclos,int neuroniosEscondidos,double taxaAprendizado,Instances data) throws Exception {
		
		Map<MyClassifier,Instances> comite = new HashMap<MyClassifier,Instances>();
		
		//Construir classificadores
		for(int i=0;i<quantClassifier;i++) {
			ArtificialNeuralNetwork mlp = new ArtificialNeuralNetwork();
			mlp.buildClassifier(ciclos,neuroniosEscondidos,taxaAprendizado, data);
			comite.put(mlp, data);
		}
		
		return executeCommittee(comite,seed,data); 	
	}
	
	public double executeCommittee(Class<? extends DecisionTree> cClass, int seed,int quantClassifier,boolean unpruned, Instances data) throws Exception {
		
		Map<MyClassifier,Instances> comite = new HashMap<MyClassifier,Instances>();
		
		//Construir classificadores
		for(int i=0;i<quantClassifier;i++) {
			DecisionTree dt = new DecisionTree();
			dt.buildClassifier(unpruned, data);
			comite.put(dt, data);
		}
		
		return executeCommittee(comite,seed,data); 	
	}
	
	public double executeCommittee(Class<? extends KNearestNeighbors> cClass, int seed,int quantClassifier,int k,SelectedTag weight,Instances data) throws Exception {
		
		Map<MyClassifier,Instances> comite = new HashMap<MyClassifier,Instances>();
		
		//Construir classificadores
		for(int i=0;i<quantClassifier;i++) {
			KNearestNeighbors knn = new KNearestNeighbors();
			knn.buildClassifier(k,weight,data);
			comite.put(knn, data);
		}
		
		return executeCommittee(comite,seed,data); 	
	}
	
	public double executeCommittee(Class<? extends NaiveBayes> cClass, int seed,int quantClassifier, Instances data) throws Exception {
		
		Map<MyClassifier,Instances> comite = new HashMap<MyClassifier,Instances>();
		
		//Construir classificadores
		for(int i=0;i<quantClassifier;i++) {
			experimento.weka.naosupervisionados.NaiveBayes nb = new experimento.weka.naosupervisionados.NaiveBayes();
			nb.buildClassifier(data);
			comite.put(nb, data);
		}
		
		return executeCommittee(comite,seed,data); 	
	}
	
	private double executeCommittee(Map<MyClassifier,Instances> comite, int seed, Instances data) throws Exception {
		
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
			
			if(maioriaVotos == i.classValue()) {
				pctCorrect+=(1/data.size());
			}
			
			//Limpar valores
			for(int v=0;v<data.numClasses();v++) {
				classifications.set(v,0.0);
			}
		}
		
		return pctCorrect;
		
	}
	
	
}
