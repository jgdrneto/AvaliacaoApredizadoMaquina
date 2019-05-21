package experimento.weka.supervisionados;

import java.util.ArrayList;
import java.util.List;

import weka.clusterers.HierarchicalClusterer;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.jaxbbindings.Attribute;

public class HA extends MLClusterer{
	
	HierarchicalClusterer clusterer;
		
	public HA() {
		clusterer = new weka.clusterers.HierarchicalClusterer();
		this.setClusterer(clusterer);
		this.setDistancefunction((new Kmeans()).getDistancefunction());
	}

	@Override
	protected void preProcess(int seed, int nGroups) {
		try {
			this.clusterer.setNumClusters(nGroups);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	protected Instance obterCentroide(int numbClass) {
		return this.centroides.get(numbClass);
	}
	
}
