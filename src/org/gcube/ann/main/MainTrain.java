package org.gcube.ann.main;

import java.io.File;

import org.gcube.ann.feedforwardann.DichotomicANN;

public class MainTrain {

	public static void main(String[] args){
		
		String trainingFile = new File("sample_ANDport.csv").getAbsolutePath();
		//String trainingFile = new File("sample_XORport.csv").getAbsolutePath();
		double  learningThreshold = 0.0001;
		int numberOfCycles = 1000;
		//String layerS = "0";
		//String layerS = "20|10|5";
		String layerS = "20|10";
		if (args!= null && args.length>0) {
			trainingFile = args[0];
			learningThreshold = Double.parseDouble(args[1]);
			numberOfCycles=Integer.parseInt(args[2]);
			layerS = args[3];
		}
		String [] layerSs = layerS.split("\\|");
		
		int[] layers=new int[layerSs.length]; 
		
		for (int j=0;j<layers.length;j++)
			layers[j] = Integer.parseInt(layerSs[j]);

		DichotomicANN trainer = new DichotomicANN();
		
		String targetColumn = "t";
		String inputColumns [] = {"a","b"};
		
		trainer.train(trainingFile, learningThreshold, numberOfCycles, layers, inputColumns, targetColumn);
		
	}
	
	
	
}
