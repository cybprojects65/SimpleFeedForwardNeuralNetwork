package org.gcube.ann.main;

import java.io.File;

import org.gcube.ann.feedforwardann.DichotomicANN;

public class MainTest {

	public static void main(String[] args){
		
		String testFile = new File("sample_ANDport.csv").getAbsolutePath();
		String annFile = new File("sample_ANDport.ann").getAbsolutePath();
		//String testFile = new File("sample_XORport.csv").getAbsolutePath();
		//String annFile = new File("sample_XORport.ann").getAbsolutePath();
		
		if (args!= null && args.length>0) {
			testFile = args[0];
			annFile = args[1];
		}
		DichotomicANN trainer = new DichotomicANN();
		
		String inputColumns [] = {"a","b"};
		
		trainer.test(testFile, annFile, inputColumns);
		
	}
	
	
	
}
