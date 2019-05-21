package experimento.weka.supervisionados;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.core.AttributeStats;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public abstract class MLClusterer {
	DistanceFunction distancefunction;
	Instances base;
	Instances baseNotClass;
	List<Group> grupos;
	AbstractClusterer clusterer;
	List<Instance> centroides; 
	String csvString;
	
	public MLClusterer() {
		csvString = this.getCSVHearder()+"\n";
	}
	
	public void init(int numGroups) {
		
		this.grupos = new ArrayList<Group>();
		
		for(int i=0;i<numGroups;i++) {
			grupos.add(new Group());
		}
		this.centroides = new ArrayList<Instance>();
	}
	
	public Clusterer getClusterer() {
		return clusterer;
	}
	
	public void setClusterer(AbstractClusterer clusterer) {
		this.clusterer = clusterer;
	}

	public DistanceFunction getDistancefunction() {
		return distancefunction;
	}

	public void setDistancefunction(DistanceFunction distancefunction) {
		this.distancefunction = distancefunction;
	}

	public List<Instance> getCentroids() {
		
		List<Instance> result = new ArrayList<Instance>();
		
		for(Group g : this.grupos) {
			if(!g.getInstances().isEmpty()) {
				Instance media = (Instance)g.getInstances().get(0).copy();
				
				for(int a=0; a<g.getInstances().get(0).numAttributes();a++) {
					double d = 0;
					for(Instance i : g.getInstances()) {
						d+=i.value(a);
					}
					d=d/g.getInstances().size();
					media.setValue(a,d);
				}
				
				Instance minInst = g.getInstances().get(0);
				double min = MLClusterer.distance(g.getInstances().get(0),media);
				for(Instance i : g.getInstances()) {
					double value = MLClusterer.distance(media, i); 
					if(value<min) {
						minInst = i;
						min = value;
					}
				}
				
				result.add(minInst);
			}
		}
		return result;
	}
	
	public double getSilhouette() {
		
		SilhouetteIndex s = new SilhouetteIndex();
		
		try {
			s.evaluate(clusterer, this.getCentroids(), this.baseNotClass, this.distancefunction);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return s.getGlobalSilhouette();
	}
	
	public double getDavidBouldin() {
        int numberOfClusters = grupos.size();
        double david = 0.0;
        
        if (numberOfClusters == 1) {
            throw new RuntimeException(
                    "Impossible to evaluate Davies-Bouldin index over a single cluster");
        } else {
            double[] withinClusterDistance = new double[numberOfClusters];

            int i = 0;
            for (Group cluster : grupos) {
                for (Instance punto : cluster.getInstances()) {
                    withinClusterDistance[i] += MLClusterer.distance(punto, cluster.getCentroide());
                }
                withinClusterDistance[i] /= cluster.getInstances().size();
                i++;
            }


            double result = 0.0;
            double max = Double.NEGATIVE_INFINITY;

            try {
                for (i = 0; i < numberOfClusters; i++) {
                    if (grupos.get(i).getCentroide() != null) {
                        for (int j = 0; j < numberOfClusters; j++)
                            if (i != j && grupos.get(j).getCentroide() != null) {
                                double val = (withinClusterDistance[i] + withinClusterDistance[j])
                                        / MLClusterer.distance(grupos.get(i).getCentroide(), grupos.get(j).getCentroide());
                                if (val > max)
                                    max = val;
                            }
                    }
                    result = result + max;
                }
            } catch (Exception e) {
                System.out.println("Excepcion al calcular DAVID BOULDIN");
                e.printStackTrace();
            }
            david = result / numberOfClusters;
        }

        return david;
    }
		
	protected abstract void preProcess(int seed, int nGroups);
	
	public String getCSVHearder() {
		return "MODEL,K,SEED,SILHOUETTE,DB,CR"; 
	}
	
	private void addCSVString(int seed) {
		this.csvString+=this.getClass().getSimpleName()+","+
						this.grupos.size()+","+
						seed+","+
						this.getSilhouette()+","+
						this.getDavidBouldin()+","+
						this.getCR()+"\n";
	}
	
	public void toSaveCSVFile(int seed,String filename) {
		
		this.addCSVString(seed);
		
		File file = new File(filename);
		
		if(file.getParentFile()!=null) {
			file.getParentFile().mkdirs();
		}
		
		try {
			Writer writer = new FileWriter(file);
			
			writer.write(csvString);
					
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public void executeClustering(int nGroups,int seed,Instances base, Instances baseNotClass) throws Exception {
		
		/*
		Filter filterNorm = new Normalize();
        filterNorm.setInputFormat(base);
        base = Filter.useFilter(base, filterNorm);
        
        filterNorm.setInputFormat(baseNotClass);
        baseNotClass = Filter.useFilter(baseNotClass, filterNorm);
        */
		this.base =  base;
		this.baseNotClass = baseNotClass;
		
		this.init(nGroups);
		
		this.preProcess(seed,nGroups);
		
		this.clusterer.buildClusterer(baseNotClass);
		
		this.generateStruture(baseNotClass);
	
	}
	
	private void generateStruture(List<Instance> result) {
		try {
			for (Instance i : result) {
				grupos.get(this.clusterer.clusterInstance(i)).getInstances().add(i);
			}
			
			this.centroides = this.getCentroids();
			
			for(Instance i : result) {
				//if the centroid is not set
				if (grupos.get(this.clusterer.clusterInstance(i)).getCentroide() == null) {
					grupos.get(this.clusterer.clusterInstance(i)).setCentroide(obterCentroide(this.clusterer.clusterInstance(i)));
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
	}

	protected abstract Instance obterCentroide(int numbClass);
	
	public double getCR(){
		
		return this.getARI(base,baseNotClass);
	}
	
	double getARI(Instances X, List<Instance> Y) {
		
		// Tabela de contigencia
		int[][] table = new int[X.numClasses() + 1][grupos.size() + 1];

		// Numero de objetos
		int numInstances = X.numInstances();

		// Popula a tabela
		int xClass, yClass;
		for (int i = 0; i < numInstances; i++) {
			try {
				xClass = (int) X.instance(i).classValue();
				yClass = (int) this.clusterer.clusterInstance(Y.get(i));
				table[xClass][yClass]++;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		for (int i = 0; i < table.length - 1; i++) {
			for (int j = 0; j < table[i].length - 1; j++) {
				table[table.length - 1][j] += table[i][j]; // Computa a Ãºltima
															// linha
				table[i][table[i].length - 1] += table[i][j]; // Computa a
																// Ãºltima coluna
			}
		}

		double TERMO_A = 0;
		for (int i = 0; i < table.length - 1; i++) {
			for (int j = 0; j < table[i].length - 1; j++) {
				TERMO_A += Mathematics.combinationOf(table[i][j], 2);
			}
		}

		double TERMO_B = 0;
		double TERMO_C = 0;
		for (int i = 0; i < table.length - 1; i++) {
			TERMO_B += Mathematics.combinationOf(
					table[i][table[i].length - 1], 2); // Ultima coluna
			// System.out.printf("B ~> %d-%d\n", i, table[i].length - 1);
		}
		
		for (int i = 0; i < table[table.length-1].length - 1; i++) {
			TERMO_C += Mathematics.combinationOf(table[table.length - 1][i],
					2); // Ultima linha
			// System.out.printf("C ~> %d-%d\n", table.length - 1, i);
		}
				
		double TERMO_D = Mathematics.combinationOf(numInstances, 2);

		double INDEX = TERMO_A;
		double EXP_INDEX = (TERMO_B * TERMO_C) / TERMO_D;
		double MAX_INDEX = 0.5 * (TERMO_B + TERMO_C);

//		print(table);

		return ((INDEX - EXP_INDEX) / (MAX_INDEX - EXP_INDEX));
	}
	
	public static double distance(Instance x, Instance y) {
		if (x.numAttributes() != y.numAttributes())
	        throw new IllegalArgumentException(String.format("Arrays have different length: x[%d], y[%d]", x.numAttributes(), y.numAttributes()));

	    int n = x.numAttributes();
	    int m = 0;
	    double dist = 0.0;
	    for (int i = 0; i < n; i++) {
	    	if (!Double.isNaN(x.value(0)) && !Double.isNaN(y.value(0))) {
	    		m++;
	            double d = x.value(i) - y.value(i);
	            dist += d * d;
	        }
	    }

	    if (m == 0)
	    	dist = Double.NaN;
	    else
	    	dist = n * dist / m;
	    
	    return Math.sqrt(dist);
	}
	
	public double getS() {
		
		List<Double> s = new ArrayList<Double>();
		
		//Calcular somatorio de a(i)
		
		for(Group g : this.grupos) {
			double a=0,b=0;
			
			for(Instance i : g.getInstances()) {
				for(Instance j : g.getInstances()) {
					if(i.equals(j)) {
						continue;
					}else {
						a+= MLClusterer.distance(i, j);
					}	
				}
			}
			
			a=a/g.size()-1;
			
			double min = Double.NEGATIVE_INFINITY;
			for(Instance i : g.getInstances()) {
				for(Group m : this.grupos) {
					if(!g.equals(m)) {
						for(Instance j : m.getInstances()) {
							b+=MLClusterer.distance(i, j);
						}	
					}
					b = b/m.size();
				}
				if(b<min) {
					min = b;
				}
			}
			
		
		return 0;
	}	

}
