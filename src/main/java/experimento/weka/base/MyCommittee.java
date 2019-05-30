package experimento.weka.base;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.List;

import experimento.weka.base.DataSet.DATAVALUE;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public abstract class MyCommittee extends MyClassifier{
	
	protected int quant;
	protected Class<? extends MyClassifier> cClass;
	protected DecimalFormat df;
	
 	protected String csvString;

	public void resetCsvString() {
		this.csvString = this.toCSVHeader() + "\n";
	}

	public MyCommittee() {
		this.csvString=this.toCSVHeader() + "\n";
		this.df = new DecimalFormat("0.0000");
	}
	
	public MyCommittee(DecimalFormat nDf) {
		this.csvString=this.toCSVHeader() + "\n";
		this.df = nDf;
	}
	
	public abstract Evaluation evaluateClassifier(int folds,List<MyClassifier> myclassifier,int quantClassifiers, int seed, Instances data,DATAVALUE datavalue) throws Exception;
	
	@Override
	public String toCSVHeader() {
		return "COMMITTEE,CLASSIFIER,FOLDS,SEED,QUANT,DATABASE,CORRECTLY CLASSIFIED,INCORRECTLY CLASSIFIED";
	}
	
	@Override
	public String toCSV(Evaluation e, int seed, int folds, DATAVALUE dataValue) {
		
		this.csvString += 	this.getClass().getSimpleName() + "," +
							this.cClass.getSimpleName() + "," + 
							folds + "," +
							seed + "," +
							this.quant+ "," +
							dataValue.name() + "," +
							this.df.format(e.pctCorrect()) + "," +
							this.df.format(e.pctIncorrect()) + "\n";
		
		return this.csvString;
	}
	
	public void toSaveCSVFile(Evaluation eval, int seed, int folds, DATAVALUE dataValue,String filename) {
		
		this.toCSV(eval, seed,folds,dataValue);
		
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

	public int getQuant() {
		return this.quant;
	}

	public Class<? extends MyClassifier> getcClass() {
		return this.cClass;
	}

}
