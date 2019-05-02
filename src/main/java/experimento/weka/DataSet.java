package experimento.weka;

import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class DataSet{
	
	public enum DATAVALUE{
		NORMALIZED,
		NOT_NORMALIZED
	}
	
	private Instances data;

	public DataSet(String filename, DATAVALUE datavalue) throws Exception{
		
		DataSource source = new DataSource(filename);
		this.data = source.getDataSet();
		this.data.deleteStringAttributes();
		this.data.setClassIndex(data.numAttributes() - 1);
		
		if(datavalue==DATAVALUE.NORMALIZED) {
			Filter filterNorm = new Normalize();
	        filterNorm.setInputFormat(this.data);
	        this.data = Filter.useFilter(data, filterNorm);
		}
	}
	
	public Instances getData() {
		return this.data;
	}

}
