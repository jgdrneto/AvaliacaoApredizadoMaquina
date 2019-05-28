package experimento.weka.supervisionados;

import experimento.weka.base.MyClusterer;
import weka.clusterers.HierarchicalClusterer;

public class HA extends MyClusterer{
	
	HierarchicalClusterer clusterer;
		
	public HA() {
		clusterer = new weka.clusterers.HierarchicalClusterer();
		this.setClusterer(clusterer);
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
	protected void posProcess() {
		//NULL
	}

}
