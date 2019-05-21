package experimento.weka.supervisionados;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class Kmeans extends MLClusterer{
	
	SimpleKMeans kmeans;
	
	public Kmeans() {
		kmeans = new SimpleKMeans();
		this.setClusterer(kmeans);
		this.setDistancefunction(kmeans.getDistanceFunction());
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
	protected Instance obterCentroide(int numClass) {
		return kmeans.getClusterCentroids().get(numClass);
	}

	@Override
	public Instances getCentroids() {
		return this.kmeans.getClusterCentroids();
	}
	
}
