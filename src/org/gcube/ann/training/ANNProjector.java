package org.gcube.ann.training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.gcube.ann.feedforwardann.Neural_Network;

public class ANNProjector {
	
	public static void main(String [] args){
		ANNProjector simulator = new ANNProjector("test.net");
		simulator.simulate("sim.csv");
	}
	
	Neural_Network neuralnet;
	
	public ANNProjector(String annfile){
		neuralnet = Neural_Network.loadNN(annfile);
	}
	
	List<double[]> inputs;
	public void readTrainingFile(boolean hasheader, String file) throws Exception{
		
		BufferedReader bf = new BufferedReader(new FileReader(new File(file)));
		if (hasheader)
			bf.readLine();
		
		String line = bf.readLine();
		inputs = new ArrayList<double[]>();
				
		while (line != null) {
			String[] linesplit = line.split(",");
			double[] input = new double[linesplit.length];

			for (int i=0;i<linesplit.length;i++)
				input[i] = Double.parseDouble(linesplit[i]);
			
			inputs.add(input);
			
			line = bf.readLine();
		}
		
		bf.close();
	}

	public void simulate(String inputsFile){
		try{
			readTrainingFile(true, inputsFile);
			for (double[] in:inputs){
//				double out = neuralnet.getCorrectValueFromOutput(neuralnet.propagate(in)[0]);
				double out = neuralnet.propagate(in)[0];
				System.out.println(in[0]+"->"+out);
			}
		}catch(Exception e){
			e.printStackTrace();
			System.out.println("Error during the FF-ANN Training");
		}
	}
	
}
