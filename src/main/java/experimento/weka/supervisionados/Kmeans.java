package experimento.weka.supervisionados;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class Kmeans extends MLClusterer{
	
	SimpleKMeans kmeans;
	
	public Kmeans() {
		kmeans = new SimpleKMeans();
		this.setClusterer(kmeans);
	}
	
	@Override
	protected void preProcess(int seed, int groups) {
		kmeans.setSeed(seed);
		kmeans.setPreserveInstancesOrder(true);
		try {
			kmeans.setNumClusters(groups);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void posProcess() {
		
		Instances centroides = this.kmeans.getClusterCentroids();
		
		for(int i=0;i<this.getGroups().size();i++) {
			this.getGroups().get(i).setCentroid(centroides.get(i));
		}
	}
}
