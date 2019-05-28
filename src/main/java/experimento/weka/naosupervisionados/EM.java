package experimento.weka.naosupervisionados;

import experimento.weka.base.MyClusterer;

public class EM extends MyClusterer{
	
	weka.clusterers.EM clusterer;
		
	public EM() {
		clusterer = new weka.clusterers.EM();
		this.setClusterer(clusterer);
	}

	@Override
	protected void preProcess(int seed, int nGroups) {
		this.clusterer.setSeed(seed);
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
