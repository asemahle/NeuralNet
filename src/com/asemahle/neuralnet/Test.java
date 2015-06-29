package com.asemahle.neuralnet;

public class Test {

	public static void main(String[] args) throws Exception 
	{
		Sigmoid s = new Sigmoid();
		NeuralNet nn = new NeuralNet(2, 1, 2, 2, false, s);
		nn.initConnectionWeights(0, 0.1, 0, 0.1);
		
		nn.mutate(0.01, 0.01);		
		double[] i = {1,2};
		double[] output = nn.input(i);
		System.out.println(output[0]);
		
		NeuralNet[] nns = {nn, nn};
		NeuralNet.saveToFile("savefile", nns);
		
		NeuralNet[] lnns = NeuralNet.loadFromFile("savefile", s);
		output = lnns[1].input(i);
		System.out.println(output[0]);
		
		NeuralNet ne = nn.copy();
		output = ne.input(i);
		System.out.println(output[0]);
		
		System.out.println(ne.getNumInputs());
		System.out.println(ne.getNumOutputs());
		
	}
}
