
public class Sigmoid implements ActivationFunction {
	
	public Sigmoid() {}
	
	public double call(double d)
	{
		double result = 1.0 / (1.0 + Math.exp(-d));
		return result;
	}
	
}
