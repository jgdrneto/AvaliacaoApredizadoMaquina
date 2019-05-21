package experimento.weka.supervisionados;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Group {
	
    private List<Instance> instancias;
    private Instance centroide;
    
    public Group() {
		this.instancias =  new ArrayList<Instance>();
	}
    
    public Instance getCentroide() {
        return centroide;
    }

    public void setCentroide(Instance centroide) {
        this.centroide = centroide;
    }

    public List<Instance> getInstances() {
        return instancias;
    }

	public double size() {
		return this.instancias.size();
	}
}
