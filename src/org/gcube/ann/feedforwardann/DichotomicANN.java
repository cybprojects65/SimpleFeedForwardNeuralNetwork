package org.gcube.ann.feedforwardann;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class DichotomicANN {

	public List<Double> targets;
	public List<Double[]> inputs;

	public Double[] mininputs;
	public Double[] maxinputs;

	public Double minTarget = Double.MAX_VALUE;
	public Double maxTarget = -Double.MAX_VALUE;

	public String header = "";
	public File outputANN;
	public File outputTrainingProjection;
	public File outputTestProjection;
	
	public double averageAbsoluteTrainingError = 0d;
	
	public void readTrainingFile(String file, String targetColumn, String[] inputColumns) throws Exception {

		List<String> allLines = Files.readAllLines(new File(file).toPath());
		String header[] = allLines.get(0).split(",");
		int outputIdx = 0;
		int counter = 0;
		List<Integer> inputIdxs = new ArrayList<Integer>();
		List<String> inCols = new ArrayList<String>(Arrays.asList(inputColumns));
		this.header = "";
		for (String h : header) {

			if (h.equals(targetColumn)) {
				outputIdx = counter;
			}

			if (inCols.contains(h)) {
				inputIdxs.add(counter);
				this.header += h + ",";
			}

			counter++;
		}
		this.header += targetColumn;

		targets = new ArrayList<Double>();
		inputs = new ArrayList<Double[]>();

		mininputs = new Double[inputIdxs.size()];
		for (int i = 0; i < mininputs.length; i++)
			mininputs[i] = Double.MAX_VALUE;

		maxinputs = new Double[inputIdxs.size()];
		for (int i = 0; i < maxinputs.length; i++)
			maxinputs[i] = 0d;
		
		int nlines = allLines.size();

		for (int i = 1; i < nlines; i++) {
			String[] linesplit = allLines.get(i).split(",");
			Double[] input = new Double[inputIdxs.size()];
			int ctr = 0;
			for (Integer inputIdx : inputIdxs) {
				Double dinput = Double.parseDouble(linesplit[inputIdx]);
				Double min = mininputs[ctr];
				Double max = maxinputs[ctr];

				if (dinput > max)
					maxinputs[ctr] = dinput;
				if (dinput < min)
					mininputs[ctr] = dinput;
				
				input[ctr] = dinput;
				ctr++;
			}

			Double dtarget = Double.parseDouble("" + linesplit[outputIdx]);
			if (dtarget > maxTarget)
				maxTarget = dtarget;
			if (dtarget < minTarget)
				minTarget = dtarget;

			targets.add(dtarget);
			inputs.add(input);

		}
		System.out.println("Max inputs: " + Arrays.toString(maxinputs));
		System.out.println("Min inputs: " + Arrays.toString(mininputs));
	}


	public void readTestFile(String file, String[] inputColumns) throws Exception {
		List<String> allLines = Files.readAllLines(new File(file).toPath());
		String header[] = allLines.get(0).split(",");
		int counter = 0;
		List<Integer> inputIdxs = new ArrayList<Integer>();
		List<String> inCols = new ArrayList<String>(Arrays.asList(inputColumns));
		this.header = "";
		for (String h : header) {

			if (inCols.contains(h)) {
				inputIdxs.add(counter);
				this.header += h + ",";
			}
			counter++;
		}
		
		inputs = new ArrayList<Double[]>();
		int nlines = allLines.size();
		for (int i = 1; i < nlines; i++) {
			String[] linesplit = allLines.get(i).split(",");
			Double[] input = new Double[inputIdxs.size()];
			int ctr = 0;
			for (Integer inputIdx : inputIdxs) {
				Double dinput = Double.parseDouble(linesplit[inputIdx]);
				input[ctr] = dinput;
				ctr++;
			}
			inputs.add(input);
		}
	}


	public void test(String testSetFiles, String preTrainedANN, String[] inputColumns) {
		
		try {
			
			outputTestProjection  = new File(testSetFiles.replace(".csv", "_testProjected.csv"));
			System.out.println("Reading training file");
			readTestFile(testSetFiles, inputColumns);
			
			int numbOfData = inputs.size();
			int numberOfInputNodes = inputs.get(0).length;
			
			Neural_Network nn = Neural_Network.loadNN(preTrainedANN);

			double[][] in = new double[numbOfData][numberOfInputNodes];
			for (int i = 0; i < numbOfData; i++) {
				Double inputarray [] = inputs.get(i);
				for (int j = 0; j < numberOfInputNodes; j++) {
					double inputi = Neural_Network.preprocess(inputarray[j]);
					in[i][j] = nn.getNormalValue(inputi, j);
				}
			}
			
			System.out.println("FF-ANN - Testing");
			List<String> allLines = Files.readAllLines(new File(testSetFiles).toPath());
			FileWriter fw = new FileWriter(outputTestProjection);
			System.out.println("FF-ANN - Projecting into the file: " + outputTestProjection.getAbsolutePath());
			fw.write(allLines.get(0)+ ","+"ANN_estimated\n");
			DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
			DecimalFormat form = new DecimalFormat("0.00",symbols);
			
			for (int i = 0; i < numbOfData; i++) {
				double [] output = nn.propagate(in[i]);
				double realoutput = nn.getCorrectValueFromOutput(output[0]);
				//System.out.println(Arrays.toString(in[i])+"->"+form.format(realoutput));
				fw.write(allLines.get(i+1) +","+ form.format(realoutput) + "\n");
			}
			fw.close();
			System.out.println("FF-ANN - Done!");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("FF-ANN - ERROR during training");
		}

	}
	
	public void train(String trainingSetFiles, double learningthreshold, int numberofcycles, 
			int[] innerlayers, String[] inputColumns, String targetColumn) {
		train(trainingSetFiles, learningthreshold, numberofcycles, 0.5f,
				innerlayers, inputColumns, targetColumn);
	}
	
	public void train(String trainingSetFiles, double learningthreshold, int numberofcycles, float learningRate,
			int[] innerlayers, String[] inputColumns, String targetColumn) {
		try {
			outputANN = new File(trainingSetFiles.replace(".csv", ".ann"));
			outputTrainingProjection  = new File(trainingSetFiles.replace(".csv", "_trainingReprojected.csv"));
			
			System.out.println("Reading training file");
			readTrainingFile(trainingSetFiles, targetColumn, inputColumns);
			int numbOfData = targets.size();

			// setup Neural Network
			int numberOfInputNodes = inputs.get(0).length;
			int numberOfOutputNodes = 1;
			Neural_Network nn;

			System.out.println("FF-ANN - Training the FF-ANN with " + numbOfData + " training data and "
					+ numberOfInputNodes + " inputs...");

			if (innerlayers != null && (innerlayers[0] > 0)) {
				System.out.println("FF-ANN - Setting up inner layers");
				int[] innerLayers = Neural_Network.setupInnerLayers(innerlayers);
				nn = new Neural_Network(numberOfInputNodes, numberOfOutputNodes, innerLayers,
						Neural_Network.ACTIVATIONFUNCTION.SIGMOID);
			} else
				nn = new Neural_Network(numberOfInputNodes, numberOfOutputNodes,
						Neural_Network.ACTIVATIONFUNCTION.SIGMOID);

			nn.maxfactor = maxTarget;
			nn.minfactor = minTarget;
			nn.mininputs = mininputs;
			nn.maxinputs = maxinputs;
			
			nn.setThreshold(learningthreshold);
			nn.setCycles(numberofcycles);
			nn.setLearningRate(learningRate);
			
			System.out.println("FF-ANN - parameters: M: " + nn.maxfactor + ", m: " + nn.minfactor + ", lt: "
					+ learningthreshold + ", it: " + numberofcycles);
			System.out.println("FF-ANN - topology: " + nn.griglia.length + " layers");
			System.out.println("FF-ANN - Now Preprocessing Features");
			
			double[][] in = new double[numbOfData][numberOfInputNodes];
			double[][] out = new double[numbOfData][numberOfOutputNodes];
			// build NN input
			for (int i = 0; i < numbOfData; i++) {
				double outputi = Neural_Network.preprocess(targets.get(i));
				out[i][0] = nn.getCorrectValueForOutput(outputi);
				Double inputarray [] = inputs.get(i);
				
				for (int j = 0; j < numberOfInputNodes; j++) {
					double inputi = Neural_Network.preprocess(inputarray[j]);
					in[i][j] = nn.getNormalValue(inputi, j);
				}
			}
			
			System.out.println("FF-ANN - Training");
			nn.train(in, out);
			System.out.println("FF-ANN - Training Done");

			List<String> allLines = Files.readAllLines(new File(trainingSetFiles).toPath());
			
			FileWriter fw = new FileWriter(outputTrainingProjection);
			
			System.out.println("FF-ANN - Projecting into the file: " + outputTrainingProjection.getAbsolutePath());
			fw.write(allLines.get(0)+ ","+"ANN_estimated\n");
			double totalerror = 0;
			DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
			DecimalFormat form = new DecimalFormat("0.00",symbols);
			
			for (int i = 0; i < numbOfData; i++) {
				double [] output = nn.propagate(in[i]);
				double realoutput = nn.getCorrectValueFromOutput(output[0]);
				double theoreticaloutput = targets.get(i);
				double error = Math.abs(realoutput-theoreticaloutput);
				
				//System.out.println(Arrays.toString(in[i])+"->"+form.format(realoutput)+" theor: "+theoreticaloutput+" err:"+form.format(error));
				totalerror += error;
				fw.write(allLines.get(i+1) +","+ form.format(realoutput) + "\n");
			}
			
			averageAbsoluteTrainingError = totalerror/(double) numbOfData;
			
			fw.close();
			System.out.println("FF-ANN - saving ANN to "+outputANN.getAbsolutePath());
			save(outputANN.getAbsolutePath(),nn);
			System.out.println("FF-ANN - average absolute error on training set: "+form.format(averageAbsoluteTrainingError));
			System.out.println("FF-ANN - Done!");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("FF-ANN - ERROR during training");
		}

	}

	public static void save(String nomeFile, Neural_Network nn) {

		File f = new File(nomeFile);
		FileOutputStream stream = null;
		try {
			stream = new FileOutputStream(f);
			ObjectOutputStream oos = new ObjectOutputStream(stream);
			oos.writeObject(nn);
			oos.close();
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
