
public class Instances {
	
	private double[] attributes;
	private double[] normalizedAttributes;
	private String i;

	public Instances(double[] attributes, String i){
		this.attributes = attributes;
		this.i = i;
	}
	
	//get attributes to normalize them
	public double[] normalize(){
		return attributes;
	}
	
	//set normalized attributes
	public void setNorm(double[] norm){
		normalizedAttributes = norm;
	}
	
	//retrieve attributes
	public double[] getAttributes(){
		return normalizedAttributes;
	}

	public String getI(){
		return i;
	}
}
