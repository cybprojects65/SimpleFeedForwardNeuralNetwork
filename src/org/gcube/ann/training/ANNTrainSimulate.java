package org.gcube.ann.training;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.gcube.ann.feedforwardann.Neural_Network;

public class ANNTrainSimulate {

	public static void main(String[] args){
		ANNTrainSimulate trainer = new ANNTrainSimulate();
		if (args==null || (args.length<8)){
			System.out.println("FF-ANN - Error! few arguments: expected at least 8 args:");
			System.out.println("1 - CSV file (comma separated with headers) with prior points. E.g.: priors.csv");
			System.out.println("2 - CSV file (comma separated with headers) with other training points. E.g.: training.csv");
			System.out.println("3 - Output file name. E.g.: projection.csv");
			System.out.println("4 - Weight of the prior points. E.g.: 50");
			System.out.println("5 - Weight of the other training points. E.g.: 1");
			System.out.println("6 - Learning Threshold for the FF-ANN. E.g.: 0.0000001");
			System.out.println("7 - Number of learning cycles of the FF-ANN. E.g.: 1000");
			System.out.println("8 - Number of neurons in each hidden layer (blank space separated) . E.g.: 0 or e.g.: 10 2 3");
			
			System.exit(0);
		}
		String priorsFile = args[0];
		String trainingSetFile = args[1];
		String outputFile = args[2];
		int priorsWeight=Integer.parseInt(args[3]);
		int trainingsetWeight=Integer.parseInt(args[4]);
		double  learningThreshold = Double.parseDouble(args[5]);
		int numberOfCycles=Integer.parseInt(args[6]);
		
		System.out.println("FF-ANN - Using these arguments: ");
		System.out.println("1 - CSV file (comma separated with headers) with prior points: "+priorsFile);
		System.out.println("2 - CSV file (comma separated with headers) with other training points: "+trainingSetFile);
		System.out.println("3 - Output file name: "+outputFile);
		System.out.println("4 - Weight of the prior points: "+priorsWeight);
		System.out.println("5 - Weight of the other training points: "+trainingsetWeight);
		System.out.println("6 - Learning Threshold for the FF-ANN: "+learningThreshold);
		System.out.println("7 - Number of learning cycles of the FF-ANN: "+numberOfCycles);
		System.out.println("8 - Number of neurons in each hidden layer (blank space separated): "+args[7]+"\n");
		int nli=7;
		int[] layers=new int[args.length-nli]; 
		for (int j=nli;j<args.length;j++)
			layers[j-nli] = Integer.parseInt(args[j]);

		String[] testfiles = new String[priorsWeight+trainingsetWeight];
		
		for (int i=0;i<priorsWeight;i++)
			testfiles[i] = priorsFile;

		for (int i=priorsWeight;i<(priorsWeight+trainingsetWeight);i++)
			testfiles[i] = trainingSetFile;
		
		
		trainer.train(testfiles, outputFile, learningThreshold, numberOfCycles, layers);
		System.out.println("FF-ANN - ...exiting");
	}
	
	List<Object[]> targets;
	List<Object[]> inputs;
	Double minTarget = Double.MAX_VALUE;
	Double maxTarget = -Double.MAX_VALUE;
	
	Double minInput = Double.MAX_VALUE;
	Double maxInput = -Double.MAX_VALUE;
	String header = "";
	
	
	public String mergeFiles(boolean hasheader, String... files) throws Exception{
		StringBuffer sb = new StringBuffer();
		for (String file:files){
			BufferedReader bf = new BufferedReader(new FileReader(new File(file)));
			if (hasheader)
				header = bf.readLine();
			
			String line = bf.readLine();
			while(line!=null){
				sb.append(line+"\n");
				 line = bf.readLine();
			}
			bf.close();
		}
		
		return sb.toString();
	}
	
	public void readTrainingFile(boolean hasheader, String... files) throws Exception{
		String mergedFiles = mergeFiles(hasheader, files);
		InputStream is = new ByteArrayInputStream(mergedFiles.getBytes());
		// read it with BufferedReader
		BufferedReader bf = new BufferedReader(new InputStreamReader(is));
		
		String line = bf.readLine();
		targets = new ArrayList<Object[]>();
		inputs = new ArrayList<Object[]>();
				
		while (line != null) {
			String[] linesplit = line.split(",");
			Object[] target = new Object[1];
			Object[] input = new Object[linesplit.length-1];
			target[0] = linesplit[0];
			Double dtarget = Double.parseDouble(linesplit[0]);
			Double dinput = Double.parseDouble(linesplit[1]);

			if (dinput>maxInput)
				maxInput=dinput;
			if (dinput<minInput)
				minInput=dinput;

			
			if (dtarget>maxTarget)
				maxTarget=dtarget;
			if (dtarget<minTarget)
				minTarget=dtarget;
			
			for (int i=1;i<linesplit.length;i++)
				input[i-1] = linesplit[i];
			
			
			
			targets.add(target);
			inputs.add(input);
			
			line = bf.readLine();
		}
		
		bf.close();
	}
	
	
	public double getNormalValue(double numb){
		return (double)(numb-minInput)/(maxInput-minInput);
	}
	
	public double getDeNormalValue(double numb){
			return numb*maxInput+(1-numb)*minInput;
	}
	
	
	public void train(String[] trainingSetFiles, String outputFile, double learningthreshold, int numberofcycles, int... innerlayers){
			try{
					readTrainingFile(true, trainingSetFiles);
					int numbOfFeatures = targets.size();

					// setup Neural Network
					int numberOfInputNodes = inputs.get(0).length;
					int numberOfOutputNodes = 1;
					Neural_Network nn;
					
					System.out.println("FF-ANN - Training the FF-ANN with "+numbOfFeatures+" training data and "+numberOfInputNodes+" inputs...");
					
					if (innerlayers!=null && (innerlayers[0]>0)){
						int[] innerLayers = Neural_Network.setupInnerLayers(innerlayers);
						nn = new Neural_Network(numberOfInputNodes, numberOfOutputNodes, innerLayers, Neural_Network.ACTIVATIONFUNCTION.SIGMOID);
					}
					else
						nn = new Neural_Network(numberOfInputNodes, numberOfOutputNodes, Neural_Network.ACTIVATIONFUNCTION.SIGMOID);
					
					nn.maxfactor=maxTarget;
					nn.minfactor=minTarget;
					nn.setThreshold(learningthreshold);
					nn.setCycles(numberofcycles);
					
					System.out.println("FF-ANN - parameters: M: "+nn.maxfactor+", m: "+nn.minfactor+", lt: "+learningthreshold+", it: "+numberofcycles);
					System.out.println("FF-ANN - topology: "+nn.griglia.length+"X"+nn.griglia[0].length);
					
					System.out.println("FF-ANN - Now Preprocessing Features");
					
					double[][] in = new double[numbOfFeatures][];
					double[][] out = new double[numbOfFeatures][];
					// build NN input
					for (int i = 0; i < numbOfFeatures; i++) {
						in[i] = Neural_Network.preprocessObjects(Arrays.copyOfRange((Object[]) inputs.get(i), 0, numberOfInputNodes));
						out[i] = Neural_Network.preprocessObjects(Arrays.copyOfRange((Object[]) targets.get(i), 0, numberOfOutputNodes));
						in[i][0] =getNormalValue(in[i][0]);
						out[i][0] =nn.getCorrectValueForOutput(out[i][0]); 
					}
					System.out.println("FF-ANN - Training");
					nn.train(in, out);
					double sout = 0;
					System.out.println("FF-ANN - Training Done");
					
					FileWriter fw = new FileWriter(new File(outputFile));
					fw.write(header+"\n");
					
					System.out.println("FF-ANN - Projecting into the file: "+outputFile);
					double[][] realoutputs = new double[out.length][out[0].length];
					double[][] toutputs = new double[out.length][out[0].length];
					
					for (int i=0;i <numbOfFeatures; i++){
						double output = nn.propagate(in[i])[0];
						double realoutput = nn.getCorrectValueFromOutput(output);
						double realinput =  getDeNormalValue(in[i][0]);
//						System.out.println(in[i][0]+"->"+out[i][0]+":"+output);
						sout+=(output-out[i][0])*(output-out[i][0]);
						fw.write(realoutput+","+realinput+"\n");
						realoutputs[i][0] = realoutput;
						toutputs[i][0] = nn.getCorrectValueFromOutput(out[i][0]); 
					}
					
//					showChart(in,realoutputs,toutputs,numbOfFeatures);
					fw.close();
					
					sout = Math.sqrt(sout/(double) numbOfFeatures)*100d;
//					System.out.println("FF-ANN - Avg discrepancy: "+sout);
					System.out.println("FF-ANN - Done!");
				} catch (Exception e) {
					e.printStackTrace();
					System.out.println("FF-ANN -ERROR during training");
				}
				
	}
	
	/*
	void showChart(double[][] in, double[][] output, double[][] toutput, int npoints) throws Exception {
		
		
		XYSeries neuralnetworkseries = new XYSeries("Neural Network");
		XYSeries rulebasedHS = new XYSeries("Rule-based HS");
		XYSeriesCollection xyseriescollection = new XYSeriesCollection(neuralnetworkseries);
		xyseriescollection.addSeries(rulebasedHS);
		
		for (int i=0;i <npoints; i++){
			neuralnetworkseries.add(getDeNormalValue(in[i][0]),output[i][0]);
			rulebasedHS.add(getDeNormalValue(in[i][0]),toutput[i][0]);
		}
		
		
		NumberAxis numberaxis = new NumberAxis("SSB");
		numberaxis.setAutoRangeIncludesZero(true);
		NumberAxis numberaxis1 = new NumberAxis("Recruits");
		numberaxis1.setAutoRangeIncludesZero(true);
		XYSplineRenderer xysplinerenderer = new XYSplineRenderer();
		xysplinerenderer.setDrawSeriesLineAsPath(true);
		xysplinerenderer.setSeriesStroke(0, new BasicStroke(1.5f));
		xysplinerenderer.setSeriesShapesVisible(0, false);
		xysplinerenderer.setSeriesShape(1, new Ellipse2D.Double(-3, -3, 6, 6));
		xysplinerenderer.setUseFillPaint(true); 
		xysplinerenderer.setSeriesStroke(1, new BasicStroke(0));
		
		XYPlot xyplot = new XYPlot((XYDataset) xyseriescollection, numberaxis, numberaxis1,xysplinerenderer);
		
		JFreeChart chart = new JFreeChart(xyplot);
		chart.setAntiAlias(true);
		ChartPanel panel = new ChartPanel(chart, true);
		final String headless = System.getProperty("java.awt.headless", "false");
		if (!headless.equalsIgnoreCase("true")) {
			try {
				JFrame frame = new JFrame("");
				frame.addWindowListener(new WindowAdapter() {
					public void windowClosing(WindowEvent e) {
						e.getWindow().dispose();
					}
				});

				
				frame.setContentPane(panel);
				frame.setSize(new Dimension(500, 500));
				frame.setVisible(true);
				
			} catch (HeadlessException exception) {
				
				return;
			}
		}
	}
	*/
	
	public static void save(String nomeFile, Neural_Network nn) {

		File f = new File(nomeFile);
		FileOutputStream stream = null;
		try {
			stream = new FileOutputStream(f);
			ObjectOutputStream oos = new ObjectOutputStream(stream);
			oos.writeObject(nn);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("FF-ANN - ERROR in writing object on file: " + nomeFile);
		} finally {
			try {
				stream.close();
			} catch (IOException e) {
			}
		}
		System.out.println("FF-ANN - Success in writing object on file: " + nomeFile);
	}
	
	
	
}
