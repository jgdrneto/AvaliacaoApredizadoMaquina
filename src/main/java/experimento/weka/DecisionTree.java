package experimento.weka;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTree extends J48{

	private static final long serialVersionUID = 1L;
	
	public DecisionTree() {
		super();
	}
	
	public DecisionTree(Boolean unpruned) {
		super();
		this.setUnpruned(unpruned);
	}
	
	public List<Double> classifyAll(Instances instances) throws Exception {
		
		this.buildClassifier(instances);
		
		List<Double> valores = new ArrayList<Double>();
		
		for(Instance i : instances) {
			valores.add(this.classifyInstance(i));
		}
		
		return valores;
	}
}
