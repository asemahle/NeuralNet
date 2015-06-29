package com.asemahle.neuralnet;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * A neural network. WOW. I sure hope it works.
 * 
 * @author Aidan Mahler
 *
 */
public class NeuralNet
{	
	
	public NeuralNet (
			int numInputs, 
			int numOutputs,
			int numLayers,
			int neuronsPerLayer,
			boolean hasBias,
			ActivationFunction activationFunc) throws Exception
	{
		//Ensure parameters are valid
		if (numInputs < 1) throw new Exception("numInputs must be > 0");
		if (numOutputs < 1) throw new Exception("numOutputs must be > 0");
		if (numLayers < 1) throw new Exception("numLayers must be > 0");
		if (neuronsPerLayer < 1) 
			throw new Exception("neuronsPerLayer must be > 0");
		
		this.activationFunc = activationFunc;
		this.hasBias = hasBias;
		this.weights = new double[numLayers+1][][];
		
		//Add an extra weight for bias, if toggled
		int bWeight = 0;
		if (this.hasBias)
		{
			bWeight = 1;
		}
		//weights connecting input neurons to first layer
		this.weights[0] = new double[neuronsPerLayer][numInputs + bWeight];
		for (int i = 0; i < this.weights.length-1; i++)
		{   
			//weights between layers
			this.weights[i] = new double[neuronsPerLayer][neuronsPerLayer + bWeight];
		}
		//weights between the final layer and the output neurons
		this.weights[numLayers] = new double[numOutputs][neuronsPerLayer + bWeight];
	}
	
	public int getNumInputs()
	{
		int numInputs = this.weights[0][0].length;
		if (this.hasBias)
		{
			numInputs --;
		}
		return numInputs;
	}
	
	public int getNumOutputs()
	{
		int numOutputs = this.weights[weights.length - 1].length;
		if (this.hasBias)
		{
			numOutputs --;
		}
		return numOutputs;
	}
	
	public double[][][] getWeights()
	{
		double[][][] w = NeuralNet.copyWeights(this.weights);
		return w;
	}
	
	public ActivationFunction getActivationFunc()
	{
		return this.activationFunc;
	}
	
	public boolean hasBias()
	{
		return this.hasBias;
	}
	
	public void initConnectionWeights(
			double weightMean, double weightStdev,
			double biasMean, double biasStdev)
	{	
		for (double[][] layer : this.weights)
		{
			for (double[] neuron : layer)
			{
				for (int conn = 0; conn < neuron.length; conn++)
				{
					if (this.hasBias && conn == neuron.length - 1)
					{
						//The last connection to any neuron is the Bias (if bias is toggled)
						neuron[conn] = this.rand.nextGaussian() * biasStdev + biasMean;
					}
					else
					{
						neuron[conn] = this.rand.nextGaussian() * weightStdev + weightMean;
					}
				}
			}
		}
	}
	
	public void initConnectionWeights(double weightMean, double weightStdev)
	{
		this.initConnectionWeights(weightMean, weightStdev, 0, 0);
	}
	
	public static void saveToFile(String filename, NeuralNet[] neuralNets) throws IOException
	{
		ArrayList<Double> output = new ArrayList<Double>();
		
		output.add((double)neuralNets.length); //1. Number of neural nets to save
		for(NeuralNet nn : neuralNets)
		{
			double hb = 0;
			if (nn.hasBias())
			{
				hb = 1;
			}
			output.add(hb); //2. If the net hasBias (1 or 0)
			double[][][] weights = nn.getWeights();
			output.add((double)weights.length); //3. Number of layers
			for (double[][] layer : weights)
			{
				output.add((double)layer.length); //4. Size of the layer
				for (double[] neuron : layer)
				{
					output.add((double)neuron.length); //5. Neurons in the layer
					for (double conn : neuron)
					{
						output.add(conn); //6. Neuron connection weights
					}
				}
			}
		}

		//output doubles to file
		FileOutputStream fs = new FileOutputStream(filename);
		DataOutputStream stream = new DataOutputStream(fs);
		try {
			for (Double oneDoubleAtATimePleaseNoMoreNoLess : output)
			{
				stream.writeDouble(oneDoubleAtATimePleaseNoMoreNoLess);
			}
		} finally {
		    stream.close();
		}
	}

	public static NeuralNet[] loadFromFile(String filename, ActivationFunction activationFunc)
	{
		InputStream is = null;
		DataInputStream stream = null;
		ArrayList<Double> doubles = new ArrayList<Double>();
		
		//Open File
		try 
		{
			is = new FileInputStream(filename);
			stream = new DataInputStream(is);
		} 
		catch (FileNotFoundException e) 
		{
			System.out.println("Could not create InputStream for " + filename);
			e.printStackTrace();
		}
		
		//Read file into arraylist of doubles
		try 
		{
			while (stream.available() >= 8)
			{
				doubles.add(stream.readDouble());
			}
		} 
		catch (IOException e) 
		{
			System.out.println("Problem reading doubles from " + filename);
			e.printStackTrace();
		} 
		finally 
		{
		    try 
		    {
				stream.close();
			} 
		    catch (IOException e) 
		    {
				System.out.println("Could not close " + filename);
				e.printStackTrace();
			}
		}
		
		//Create Neural nets
		NeuralNet[] nets = new NeuralNet[doubles.remove(0).intValue()]; //1. Number of nets
		for (int netNumber = 0; netNumber < nets.length; netNumber++)
		{
			boolean hasBias = true; //2. hasBias (1 if yes, 0 if no)
			int hb = doubles.remove(0).intValue();
			if (hb == 0)
			{
				hasBias = false;
			}
			//3. Number of layers
			double[][][] weights = new double[doubles.remove(0).intValue()][][]; 
			for (int layer = 0; layer < weights.length; layer++)
			{
				//4. Number of neurons in the layer
				weights[layer] = new double[doubles.remove(0).intValue()][]; 
				for (int neuron = 0; neuron < weights[layer].length; neuron++)
				{
					//5. Number of connections to the neuron
					weights[layer][neuron] = new double[doubles.remove(0).intValue()]; 
					for (int conn = 0; conn < weights[layer][neuron].length; conn++)
					{
						//6. Add connection weights
						weights[layer][neuron][conn] = doubles.remove(0); 
					}
				}
			}
			nets[netNumber] = new NeuralNet(weights, hasBias, activationFunc);
		}
		return nets;
	}
	
	public static NeuralNet[] copy(NeuralNet[] neuralNets)
	{
		NeuralNet[] netCopies = new NeuralNet[neuralNets.length];
		for (int i = 0; i < netCopies.length; i++)
		{
			netCopies[i] = neuralNets[i].copy();
		}
		return netCopies;
	}
	
	public static void mutate(
			NeuralNet[] neuralNets,
			double weightStdev,
			double biasStdev)
	{
		for(NeuralNet nn : neuralNets){
			nn.mutate(weightStdev,biasStdev);
		}
	}
	
	public static void mutate(
			NeuralNet[] neuralNets,
			double weightStdev)
	{
		for(NeuralNet nn : neuralNets){
			nn.mutate(weightStdev);
		}
	}

	public NeuralNet copy()
	{
		return new NeuralNet(this.weights, this.hasBias, this.activationFunc);
	}

	public void mutate(double weightStdev, double biasStdev)
	{
		for (double[][] layer : this.weights)
		{
			for (double[] neuron : layer)
			{
				for (int conn = 0; conn < neuron.length; conn++)
				{
					if (this.hasBias && conn == neuron.length - 1)
					{
						neuron[conn] += this.rand.nextGaussian() * biasStdev;
					}
					else
					{
						neuron[conn] += this.rand.nextGaussian() * weightStdev;
					}
				}
			}
		}
	}
	
	public void mutate(double weightStdev)
	{
		this.mutate(weightStdev, 0);
	}
	
	public double[] input(double[] inputs) throws Exception
	{
		if (inputs.length != this.getNumInputs()) 
			throw new Exception("Number of inputs must match number of input neurons!\n"
					+ "The number of input neurons is " + this.getNumInputs());
					
		double[] currentLayer = inputs;	
		for (int layer = 0; layer < this.weights.length; layer++)
		{
			double[] nextLayer = new double[this.weights[layer].length];
			for (int neuron = 0; neuron < nextLayer.length; neuron++)
			{
				double sum = 0;
				for (int i = 0; i < currentLayer.length; i++)
				{
					sum += currentLayer[i] * this.weights[layer][neuron][i];
				}
				if (this.hasBias)
				{
					int len = this.weights[layer][neuron].length;
					sum += -1 * this.weights[layer][neuron][len - 1];
				}
				nextLayer[neuron] = this.activationFunc.call(sum);
			}
			currentLayer = Arrays.copyOf(nextLayer, nextLayer.length);
		}		
		return currentLayer;
	}
	
	private NeuralNet(
			double[][][] weights, 
			boolean hasBias, 
			ActivationFunction activationFunc)
	{
		this.weights = NeuralNet.copyWeights(weights);
		this.hasBias = hasBias;
		this.activationFunc = activationFunc;
	}
	
	private static double[][][] copyWeights(double[][][] weights)
	{
		double[][][] w = new double[weights.length][][];
		for (int layer = 0; layer < weights.length; layer++)
		{
			w[layer] = new double[weights[layer].length][];
			for (int neuron = 0; neuron < weights[layer].length; neuron++)
			{
				w[layer][neuron] = new double[weights[layer][neuron].length];
				for (int conn = 0; conn < weights[layer][neuron].length; conn++)
				{
					w[layer][neuron][conn] = weights[layer][neuron][conn];
				}
			}
		}
		return w;
	}
	
	private ActivationFunction activationFunc;
	private Random rand = new Random();
	private double[][][] weights;
	private boolean hasBias;
}
