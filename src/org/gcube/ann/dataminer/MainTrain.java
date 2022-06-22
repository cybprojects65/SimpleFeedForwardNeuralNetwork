package org.gcube.ann.dataminer;

import java.io.File;

import org.gcube.ann.training.ANNTrainTest;

public class MainTrain {

	public static void main(String[] args){
		
		
		ANNTrainTest trainer = new ANNTrainTest();
		
		File inputFile = new File("sample_ANDport.csv");
		File outputFile = new File("output.csv");
		double learningThreshold = 0.0001;
		int numberOfCycles = 1000;
		String neuronsperlayer = "0";//"1|1";
		
		if (args==null || (args.length<8)){
			
		}
		String [] npl = {"0"};
		
		 if (neuronsperlayer.trim().length()>0) {
			 npl = new String[1];
			 npl [0] = neuronsperlayer;
			 if (neuronsperlayer.contains("|"))
				 npl = neuronsperlayer.split("\\|");
			 
		 }

		int[] layers=new int[npl.length]; 
		for (int i=0;i<layers.length;i++) {
			layers[i]=Integer.parseInt(npl[i]);
		}
		
		String[] trainingfiles = {inputFile.getAbsolutePath()};
		
		trainer.train(trainingfiles, inputFile.getAbsolutePath(), outputFile.getAbsolutePath(), 
				learningThreshold, numberOfCycles, layers);
		
		System.out.println("FF-ANN - ...exiting");
		
	}
	
}
