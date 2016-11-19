
public class Distances {
	
	private int value;
	private double distance;
	private String attribute;

	public Distances(int value, double distance, String attribute){
		this.value = value;
		this.distance = distance;
		this.attribute = attribute;
	}
	
	public int getValue(){
		return value;
	}
	
	public double getDistance(){
		return distance;
	}
	
	public String getAttribute(){
		return attribute;
	}
}
