import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MyProgram {
	
	public static int tp,fp,tn,fn,size;
	public static int num;
	public static String currentLine,i;
	public static String[] holder;
	public static double maxA,maxB,maxC,maxD,maxE,maxF,maxG,maxH = Double.MIN_VALUE;
	public static double minA,minB,minC,minD,minE,minF,minG,minH = Double.MAX_VALUE;
	public static Map<Integer,Instances> theData = new HashMap<Integer,Instances>();
	public static ArrayList<Integer> yesInstances = new ArrayList<Integer>();
	public static ArrayList<Integer> noInstances = new ArrayList<Integer>();
	public static Map<String,Double> yesSums = new HashMap<String,Double>();
	public static Map<String,Double> noSums = new HashMap<String,Double>();
	public static Map<String,Double> maxAndMins = new HashMap<String,Double>();
	public static Map<Integer,ArrayList<Integer>> foldClass = new HashMap<Integer,ArrayList<Integer>>();
	
	public static void main(String[] args) throws IOException {
		
		num = 0;
		
		double[] attributes;
		
		//try...catch
		try{
			//create buffered reader
			BufferedReader br = new BufferedReader(new FileReader(args[0]));
			
			currentLine = br.readLine();
			
			//continue to read in until whole file read
			while(currentLine != null & currentLine != "\n"){
				
				holder = currentLine.split(",");
				
				attributes = new double[holder.length-1];
				
				size = holder.length-1;
				
				for(int i = 0; i < holder.length-1; i++){
					//set attributes
					attributes[i] = Double.parseDouble(holder[i]);
					
					//get maxes and mins for normalization
					if(!maxAndMins.containsKey("max" + i)){
						maxAndMins.put("max" + i, Double.parseDouble(holder[i]));
					} else {
						if(Double.parseDouble(holder[i]) > maxAndMins.get("max" + i)){
							maxAndMins.replace("max" + i, Double.parseDouble(holder[i]));
						}
					}
					
					if(!maxAndMins.containsKey("min" + i)){
						maxAndMins.put("min" + i, Double.parseDouble(holder[i]));
					} else {
						if(Double.parseDouble(holder[i]) < maxAndMins.get("min" + i)){
							maxAndMins.replace("min" + i, Double.parseDouble(holder[i]));
						}
					}
				}
				
				//update counter
				num++;
				
				//create the new instance and store it in the hashmap
				theData.put(num, new Instances(attributes,holder[holder.length-1]));
				
				//for naive bayes use, add the sums of each attribute as well as make an ArrayList of 
				//the instances to calculate mean and standard deviation
				if(holder[holder.length-1].equals("yes")){
					for(int j = 0; j < size; j++){
						if(yesSums.containsKey("attribute" + j)){
							yesSums.replace("attribute" + j, yesSums.get("attribute" + j) + attributes[j]);
						} else {
							yesSums.put("attribute" + j, attributes[j]);
						}
					}
					yesInstances.add(num);
				} else {
					for(int j = 0; j < size; j++){
						if(noSums.containsKey("attribute" + j)){
							noSums.replace("attribute" + j, noSums.get("attribute" + j) + attributes[j]);
						} else {
							noSums.put("attribute" + j, attributes[j]);
						}
					}
					noInstances.add(num);
				}
				
				//read in the next line
				currentLine = br.readLine();	
			}
			br.close();

		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + args[0]);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (NumberFormatException e){
			e.printStackTrace();
			System.out.println("Line number" + num);
		}	

		//normalize the data
		for(int n = 1; n <= num; n++){

			double[] attr = theData.get(n).normalize();
			
			double[] normalized = new double[attr.length];
			
			//normalize each value
			for(int x = 0; x < attr.length; x++){
				normalized[x] = normalize(attr[x], maxAndMins.get("max" + x), maxAndMins.get("min" + x));
			}
			
			//set the normalized values
			theData.get(n).setNorm(normalized);
		}
		
		
		int fold = 1;
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("folds.csv")));
		
		for(int i = 0; i < yesInstances.size(); i++){
			if(foldClass.containsKey(fold)){
				foldClass.get(fold).add(yesInstances.get(i));
			} else {
				ArrayList<Integer> theFold = new ArrayList<Integer>();
				theFold.add(yesInstances.get(i));
				foldClass.put(fold, theFold);
			}
			if(fold < 10){
				fold++;
			} else {
				fold = 1;
			}
		}
		
		for(int j = 0; j < noInstances.size(); j++){
			if(foldClass.containsKey(fold)){
				foldClass.get(fold).add(noInstances.get(j));
			} else {
				ArrayList<Integer> theFold = new ArrayList<Integer>();
				theFold.add(noInstances.get(j));
				foldClass.put(fold, theFold);
			}
			if(fold < 10){
				fold++;
			} else {
				fold = 1;
			}
		}
		
		for(int k = 1; k <= 10; k++){
			bw.write("fold" + k + "\n");
			
			ArrayList<Integer> foldValues = foldClass.get(k);
			
			for(int m = 0; m < foldValues.size(); m++){
				double[] instances = theData.get(foldValues.get(m)).getAttributes();
				String outcome = theData.get(foldValues.get(m)).getI();
				
				for(int n = 0; n < instances.length; n++){
					bw.write(instances[n] + ",");
				}
				
				bw.write(outcome + "\n");
			}
			bw.write("\n");
		}
		
		bw.close();
		
		//implement K-Nearest Neighbor or Naive Bayes on 10 folds... and then test data set
		if(args[2].equals("NB")){
			NBFold();
			NB(args[1]);
		} else {
			String split = args[2].substring(args[2].length() - 2, args[2].length());
			if(split.equals("NN")){
				int split2 = Integer.parseInt(args[2].substring(0,args[2].length()-2));
				KNNFold(split2);
				KNN(split2,args[1]);
			}
		}

		//uncomment for 10-fold cross-validation calculations print-out
		//System.out.println("TP: " + tp + "\nFP: " + fp + "\nTN: " + tn + "\nFN: " + fn);
		//System.out.println("Accuracy:" + (double)(tp+tn)/(double)(tp+tn+fp+fn));

	}
	
	//normalize data
	public static double normalize(double value, double max, double min){
		return (value - min) / (max - min);
	}
	
	//10-fold cross-validation implementation of KNN
	public static void KNNFold(int neighbors){
		
		//for each fold
		for(int i = 1; i <= 10; i++){
			
			//ArrayList of instances to be classified
			ArrayList<Integer> classify = foldClass.get(i);
			
			//for each of those instances
			for(int j = 0; j < classify.size(); j++){
				
				//create array to store the k-Nearest Neighbors
				Distances[] knn = new Distances[neighbors];
				for(int z = 0; z < knn.length; z++){
					knn[z] = new Distances(-1,Double.MAX_VALUE, "ERROR");
				}
				double worst = knn[knn.length-1].getDistance();
				
				//for each of the folds
				for(int k = 1; k <= 10; k++){
					
					//exclude the fold being tested
					if(i!=k){
						
						//get ArrayList of instances to test within that fold
						ArrayList<Integer> testing = foldClass.get(k);
						
						//for each instance in that fold, calculate the euclidean distance
						for(int m = 0; m < testing.size(); m++){
							double distance = distance(theData.get(classify.get(j)).getAttributes(),theData.get(testing.get(m)).getAttributes());
							//if the distance is smaller than the worst of the kNN, then find the appropriate index
							if(distance < worst){
								//System.out.println(classify.get(j) +" "+ distance + " " + theData.get(testing.get(m)).getI());
								int index = knn.length-1;
								for(int n = 0; n <= knn.length-1; n++){
									if(distance < knn[n].getDistance()){
										index = n;
										break;
									}
								}
								//move the instances that are now below the new instance down one spot
								for(int p = knn.length-2; p >= index; p--){
									knn[p+1] = knn[p];
								}
								//input the instance in the appropriate location and reset the worst value
								//System.out.println(testing.get(m) + " " + theData.get(testing.get(m)).getI());
								knn[index] = new Distances(testing.get(m),distance,theData.get(testing.get(m)).getI());
								worst = knn[knn.length-1].getDistance();
							}
						}
					}
				}
				
				//System.out.println(knn[0].getDistance() + " " + knn[1].getDistance() + " " + knn[2].getDistance());
				
				int yesTotal = 0;
				
				//count total number of yeses
				for(int r = 0; r < knn.length; r++){
					if(knn[r].getAttribute().equals("yes")){
						yesTotal++;
					}
				}
				
				//if .5+, classify as yes...increment tp,fp,tn,or fn respectfully
				if((double) yesTotal / (double) knn.length >= .5){
					//System.out.println("yes");
					if(theData.get(classify.get(j)).getI().equals("yes")){
						tp++;
					} else {
						fp++;
					}
				} else {
					//System.out.println("no");
					if(theData.get(classify.get(j)).getI().equals("no")){
						tn++;
					} else {
						fn++;
					}
				}
			}	
		}
	}
	
	//KNN algorithm
	public static void KNN(int neighbors, String fileName){
		//create try...catch
		try{
			//create buffered reader and read in the first line
			BufferedReader br2 = new BufferedReader(new FileReader(fileName));
			
			currentLine = br2.readLine();
			
			//continue until all data points are classified
			while(currentLine != null){
				
				holder = currentLine.split(",");
				
				double[] theTest = new double[holder.length];
				
				for(int i = 0; i < holder.length; i++){
					theTest[i] = Double.parseDouble(holder[i]);
				}
				
				//create array to store the k-Nearest Neighbors
				Distances[] knn = new Distances[neighbors];
				for(int z = 0; z < knn.length; z++){
					knn[z] = new Distances(-1,Double.MAX_VALUE, "ERROR");
				}
				double worst = knn[knn.length-1].getDistance();

				//for each training data point
				for(int i = 1; i <= num; i++){
					double distance = distance(theTest, theData.get(i).getAttributes());
					if(distance < worst){
						int index = knn.length-1;
						for(int n = 0; n <= knn.length-1; n++){
							if(distance < knn[n].getDistance()){
								index = n;
								break;
							}
						}
						for(int p = knn.length-2; p >= index; p--){
							knn[p+1] = knn[p];
						}
						knn[index] = new Distances(i,distance,theData.get(i).getI());
						worst = knn[knn.length-1].getDistance();
					}
				}
				
				int yesTotal = 0;
				
				//count total number of yeses
				for(int r = 0; r < knn.length; r++){
					if(knn[r].getAttribute().equals("yes")){
						yesTotal++;
					}
				}
				
				//if .5+, classify as yes...increment tp,fp,tn,or fn respectfully
				if((double) yesTotal / (double) knn.length >= .5){
					System.out.println("yes");
				} else {
					System.out.println("no");
				}
				
				currentLine = br2.readLine();
			}
			
			br2.close();
			
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + fileName);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (NumberFormatException e){
			e.printStackTrace();
			System.out.println("Line number" + num);
		}	
	}
	
	//implement 10-fold cross-validation for Naive Bayes
	public static void NBFold(){
		
		//for each of the folds
		for(int i = 1; i <= 10; i++){
			//get the ArrayList of the values of the values in the fold
			ArrayList<Integer> classify = foldClass.get(i);
			//initialize values and set them equal to 0
			double[] yesSums = new double[size];
			double[] noSums = new double[size];
			double yes, no;
			yes = no = 0;
			//for the entire data set, excluding the current fold, add the values to calculate mean and standard deviation
			for(int j = 1; j <= num; j++){
				if(!classify.contains(j)){
					if(theData.get(j).getI().equals("yes")){
						yes++;
						double[] theValues = theData.get(j).getAttributes();
						for(int z = 0; z < theValues.length; z++){
							yesSums[z] += theValues[z];
						}
					} else {
						no++;
						double[] theValues = theData.get(j).getAttributes();
						for(int y = 0; y < theValues.length; y++){
							noSums[y] += theValues[y];
						}
					}
				}
			}
			
			//calculate means
			double[] yesMeans = new double[yesSums.length];
			double[] noMeans = new double[noSums.length];
			
			for(int x = 0; x < yesSums.length; x++){
				yesMeans[x] = yesSums[x]/(double)yes;
				noMeans[x] = noSums[x]/(double)no;
			}
			
			//create standard error variables and sum the values to calculate standard deviation
			double[] yesSDs = new double[yesMeans.length];
			double[] noSDs = new double[noMeans.length];
			
			for(int k = 1; k <= num; k++){
				if(!classify.contains(k)){
					if(theData.get(k).getI().equals("yes")){
						double[] data = theData.get(k).getAttributes();
						for(int s = 0; s < data.length; s++){
							yesSDs[s] += Math.pow((double)data[s]-(double)yesMeans[s], 2);
						}
					} else {
						double[] data = theData.get(k).getAttributes();
						for(int t = 0; t < data.length; t++){
							noSDs[t] += Math.pow((double)data[t]-(double)noMeans[t], 2);
						}
					}
				}
			}
			
			for(int w = 0; w < yesSDs.length; w++){
				yesSDs[w] = Math.sqrt(yesSDs[w]/ ((double)yes-1));
				noSDs[w] = Math.sqrt(noSDs[w]/ ((double)no-1));
			}
			
			//calculate comparative statistics and classify
			for(int m = 0; m < classify.size(); m++){
				double[] instance = theData.get(classify.get(m)).getAttributes();
				
				double yesComp = calc(instance,yesMeans,yesSDs,yes/(double)num);
				double noComp = calc(instance,noMeans,noSDs,no/(double) num);
				
				if(yesComp >= noComp){
					//System.out.println(classify.get(m)+" " + theData.get(classify.get(m)).getI() +" yes");
					if(theData.get(classify.get(m)).getI().equals("yes")){
						tp++;
					} else {
						fp++;
					}
				} else {
					//System.out.println(classify.get(m)+" " + theData.get(classify.get(m)).getI() +" no");
					if(theData.get(classify.get(m)).getI().equals("no")){
						tn++;
					} else {
						fn++;
					}
				}
			}
		}
	}
	
	public static void NB(String fileName){
		
		double[] yesMeans = new double[size];
		double[] noMeans = new double[size];
		
		for(int i = 0; i < size; i++){
			yesMeans[i] = yesSums.get("attribute" + i) / (double)yesInstances.size();
			noMeans[i] = noSums.get("attribute" + i) / (double)noInstances.size();
		}
		
		double[] yesSDs = sds(yesInstances,yesMeans);
		double[] noSDs = sds(noInstances,noMeans);
		
		try{
			BufferedReader br3 = new BufferedReader(new FileReader(fileName));
			
			currentLine = br3.readLine();
			
			//continue until all data points are classified
			while(currentLine != null){
				
				holder = currentLine.split(",");
				
				double[] theTest = new double[holder.length];
				
				for(int i = 0; i < holder.length; i++){
					theTest[i] = Double.parseDouble(holder[i]);
				}
				
				double yesComp = calc(theTest,yesMeans,yesSDs,(double)yesInstances.size()/(double)num);
				double noComp = calc(theTest,noMeans,noSDs,(double)noInstances.size()/(double)num);
				
				if(yesComp >= noComp){
					System.out.println("yes");
				} else {
					System.out.println("no");
				}
				
				currentLine = br3.readLine();	
			}
			
			br3.close();
			
		}catch (FileNotFoundException e) {
			System.out.println("File not found: " + fileName);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (NumberFormatException e){
			e.printStackTrace();
			System.out.println("Line number" + num);
		}	
	}
	
	//returns the Euclidean distance for two instances
	public static double distance(double[] x, double[] y){
		
		double sum = 0;
		
		//sum the squared differences
		for(int i = 0; i < x.length; i++){
			sum += Math.pow(x[i] - y[i], 2);
		}
		
		//return the square root of the sum
		return Math.sqrt(sum);
	}
	
	public static double calc(double[] instance, double[] means, double[] SDs, double prob){
		double total = 1;
		for(int i = 0; i < instance.length; i++){
			total *= pdf(instance[i],means[i],SDs[i]);
		}
		return total * prob;
	}
	
	//calculate probability using probability density function
	public static double pdf(double value, double mean, double sd){
		return ((double) 1/(sd * Math.sqrt((double)2*Math.PI))) * Math.pow(Math.E, -0.5*
				Math.pow((value-mean)/sd,2));
	}
	
	public static double[] sds(ArrayList<Integer> instances, double[] means){
		
		//create sum of squared error variables and sum the values to calculate standard deviation
		double[] sd = new double[means.length];
		
		for(int i = 0; i < instances.size(); i++){
			double[] instance = theData.get(instances.get(i)).getAttributes();
			for(int j = 0; j < sd.length; j++){
				sd[j] += Math.pow(instance[j]-means[j], 2);
			}
		}
		
		for(int k = 0; k < sd.length; k++){
			sd[k] = Math.sqrt(sd[k]/((double)instances.size()));
		}
		
		return sd;
	}
	
}
