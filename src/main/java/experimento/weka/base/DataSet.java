package experimento.weka.base;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class DataSet{
	
	public enum DATAVALUE{
		NORMALIZED,
		NOT_NORMALIZED
	}
	
	private Instances data;

	public DataSet(String filename) throws Exception{
		
		DataSource source = new DataSource(filename);
		this.data = source.getDataSet();
		this.data.deleteStringAttributes();
		this.data.setClassIndex(data.numAttributes() - 1);
		
	}
	
	public Instances getData() {
		return this.data;
	}
	
	public Instances getDataNotClassifier() throws Exception {
		Remove removeFilter = new Remove();
		removeFilter.setAttributeIndices("last");
		removeFilter.setInputFormat(data);
		
		return Filter.useFilter(data, removeFilter);
	}
	
}