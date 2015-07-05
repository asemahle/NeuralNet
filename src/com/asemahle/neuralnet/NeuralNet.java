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
			ActivationFunction activationFunc)
	{
		//Ensure parameters are valid
		if (numInputs < 1) throw new RuntimeException("numInputs must be > 0");
		if (numOutputs < 1) throw new RuntimeException("numOutputs must be > 0");
		if (numLayers < 1) throw new RuntimeException("numLayers must be > 0");
		if (neuronsPerLayer < 1) 
			throw new RuntimeException("neuronsPerLayer must be > 0");
		
		this.activationFunc = activationFunc;
		this.hasBias = hasBias;
		this.weights = new double[numLayers+3][][];
		
		/*       VISUAL REPRESENTATION OF THE NEURAL NET WITHOUT BIAS
		 * --------------------------------------------------------------------
		 * With bias included, each neuron in the hidden layer would have one 
		 * connection:
		 *                         weights[layer][neuron][n]
		 * ...where n represents the number of neurons in each hidden layer.
		 * --------------------------------------------------------------------
		 * 
		 * weights[0][i-1][0]       
		 *                 |             input[0]  ...   input[i] 
		 *                 |                |               |
		 *                  --------------------------->    |
		 * weights[1][i-1][n-1]             |               |
		 *                  |               0      ...      0
         *                  |                                 \
         *                   -------------------------------->  \  
         * weights[2][n-1][n-1]                                   \             
         *                  |       0               0      ...      0 neuron[n]   
		 *                  |                                       | hiddenLayer[0]            
		 *                   ---------------------------------->    |
		 *                                                          |          
		 *                          0               0      ...      0
		 *                         ...             ...             ...
		 * weights[l+2][n-1][o-1]   0               0      ...      0 neuron[n]
		 *                    |                                   /   hiddenLayer[l]
		 *                     ------------------------------>  /
		 * weights[l+3][o-1][0]                               /
		 *                   |              0               0
		 *                   |              |               |
		 *                    ------------------------->    |
		 *                                  |               |
		 *                              output[0]  ...  output[o]
		 */                    
		
		
		//Add an extra weight for bias, if toggled
		int bias = 0;
		if (this.hasBias)
		{
		    bias = 1;
		}
		
		//weights connecting from inputs to input neurons
		this.weights[0] = new double[numInputs][1]; //No bias
		//weights connecting from input neurons to first hidden layer
		this.weights[1] = new double[numInputs + bias][neuronsPerLayer];
		for (int i = 2; i < this.weights.length-2; i++)
		{   
			//weights between layers
			this.weights[i] = new double[neuronsPerLayer + bias][neuronsPerLayer];
		}
		//weights between final hidden layer and output neurons
		this.weights[this.weights.length-2] = new double[neuronsPerLayer + bias][numOutputs];
		//final weights applied to outputs
		this.weights[this.weights.length-1] = new double[numOutputs][1]; //No bias
	}
	
	public int getNumInputs()
	{
		int numInputs = this.weights[0][0].length;
		return numInputs;
	}
	
	public int getNumOutputs()
	{
		int numOutputs = this.weights[this.weights.length - 1].length;
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
			double biasMean, double biasStdev,
			double outputMean, double outputStdev)
	{	
		for (int i = 0; i < this.weights.length; i++)
		{
			for (int j = 0; j < this.weights[i].length; j++)
			{
				for (int k = 0; k < this.weights[i][j].length; k++)
				{
					if((i != this.weights.length - 1) &&
					        (i != 0) &&
					        (this.hasBias) &&
					        (j == this.weights[i].length - 1))
					{
					    //bias neuron weights
						this.weights[i][j][k] = this.rand.nextGaussian() * biasStdev + biasMean;
					}
					else if(i == this.weights.length - 1 )
					{
					    //output neurons to output weights
					    this.weights[i][j][k] = this.rand.nextGaussian() * outputStdev + outputMean;
					}
					else
					{
					    //regular connection weights
					    this.weights[i][j][k] = this.rand.nextGaussian() * weightStdev + weightMean;
					}
				}
			}
		}
	}
	
	public void initConnectionWeights(double weightMean, double weightStdev)
	{
		this.initConnectionWeights(weightMean, weightStdev, 0, 0, 1, 0);
	}
	
	public void initConnectionWeights(
            double weightMean, double weightStdev,
            double biasMean, double biasStdev)
	{
	    this.initConnectionWeights(weightMean, weightStdev, biasMean, biasStdev, 1, 0);
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
            double biasStdev,
            double outputStdev)
    {
        for(NeuralNet nn : neuralNets){
            nn.mutate(weightStdev, biasStdev, outputStdev);
        }
    }
	
	public static void mutate(
			NeuralNet[] neuralNets,
			double weightStdev,
			double biasStdev)
	{
		for(NeuralNet nn : neuralNets){
			nn.mutate(weightStdev, biasStdev);
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

	public void mutate(double weightStdev, double biasStdev, double outputStdev)
	{
	    for (int i = 0; i < this.weights.length; i++)
        {
            for (int j = 0; j < this.weights[i].length; j++)
            {
                for (int k = 0; k < this.weights[i][j].length; k++)
                {
                    if((i != this.weights.length - 1) &&
                            (i != 0) &&
                            (this.hasBias) &&
                            (j == this.weights[i].length - 1))
                    {
                        //bias neuron weights
                        this.weights[i][j][k] += this.rand.nextGaussian() * biasStdev;
                    }
                    else if(i == this.weights.length - 1 )
                    {
                        //output neurons to output weights
                        this.weights[i][j][k] += this.rand.nextGaussian() * outputStdev;
                    }
                    else
                    {
                        //regular connection weights
                        this.weights[i][j][k] += this.rand.nextGaussian() * weightStdev;
                    }
                }
            }
        }
	}
	
	public void mutate(double weightStdev, double biasStdev)
	{
	    this.mutate(weightStdev, biasStdev, 0);
	}
	
	public void mutate(double weightStdev)
	{
		this.mutate(weightStdev, 0, 0);
	}
	
	public double[] input(double[] inputs) throws RuntimeException
	{
		if (inputs.length != this.getNumInputs()) 
			throw new RuntimeException("Number of inputs must match number of input neurons!\n"
					+ "The number of input neurons is " + this.getNumInputs());
					
		double[] neuronValues = new double[inputs.length];
		for (int i = 0; i < inputs.length; i++)
		{
		    //inputs to input neurons
		    neuronValues[i] = inputs[i] * this.weights[0][i][0];
		}
		
		for (int i = 1; i < weights.length - 1; i++)
		{
		    //between layers
		    neuronValues = this.calculateNextNeuronValues(neuronValues, this.weights[i]);
		}
		
		for (int i = 0; i < neuronValues.length; i++)
		{
		    //output neurons to output
		    neuronValues[i] *= weights[weights.length - 1][i][0];
		}
		
		return neuronValues;
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

    public static NeuralNet[] loadFromFile(String filename, ActivationFunction activationFunc) throws IOException
    {
        InputStream is = new FileInputStream(filename);
        DataInputStream stream = new DataInputStream(is);
        ArrayList<Double> doubles = new ArrayList<Double>();
        
        while (stream.available() >= 8)
        {
            doubles.add(stream.readDouble());
        }
        stream.close();
        
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
	
	private NeuralNet(
			double[][][] weights, 
			boolean hasBias, 
			ActivationFunction activationFunc)
	{
		this.weights = NeuralNet.copyWeights(weights);
		this.hasBias = hasBias;
		this.activationFunc = activationFunc;
	}
	
	private double[] calculateNextNeuronValues(double[] neuronValues, double[][] weights)
	{
	    double[] nextNeuronValues = new double[weights[0].length];
	    for (int i = 0; i < neuronValues.length; i++)
	    {
	        for (int j = 0; j < nextNeuronValues.length; j++)
	        {
	            nextNeuronValues[j] += weights[i][j] * neuronValues[i];
	        }
	    }

        for (int i = 0; i < nextNeuronValues.length; i++)
        {
            if (this.hasBias)
            {
                nextNeuronValues[i] += weights[weights.length - 1][i] * -1;
            }
            nextNeuronValues[i] = this.activationFunc.call(nextNeuronValues[i]);
        }
        return nextNeuronValues;
	    
	}
	
	private static double[][][] copyWeights(double[][][] weights)
	{
		double[][][] w = new double[weights.length][][];
		for (int i = 0; i < weights.length; i++)
		{
			w[i] = new double[weights[i].length][];
			for (int j = 0; j < weights[i].length; j++)
			{
				w[i][j] = new double[weights[i][j].length];
				for (int k = 0; k < weights[i][j].length; k++)
				{
					w[i][j][k] = weights[i][j][k];
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
