package experimento.weka.base;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;

public class Group {
	
    private List<Instance> instancias;
    private Instance centroid;
    
    public Group() {
		this.instancias =  new ArrayList<Instance>();
	}
    
    public Instance getCentroid() {
        return centroid;
    }

    public void setCentroid(Instance nCentroid) {
        this.centroid = nCentroid;
    }

    public List<Instance> getInstances() {
        return instancias;
    }

	public int size() {
		return this.instancias.size();
	}

}
