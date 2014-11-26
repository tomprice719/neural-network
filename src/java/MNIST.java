import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;
import au.com.bytecode.opencsv.CSVReader;

public class MNIST {
	//Class for retrieving samples from MNIST database
	//When initialize() is called, it loads all samples into memory from CSV files
	//Training and test samples can then be retrieved via getTraining and getTest
	
	Sample[] training = null;
	Sample[] test = null;
	Random rand = new Random();

	public class Sample {
		public Vector pixels;
		public Vector labels;
		public int labelIndex;
		public Sample(double[] pixels, int labelIndex) {
			this.pixels = new BasicVector(pixels);
			this.labelIndex = labelIndex;

		    double [] labelsArray  = new double[10];
		    labelsArray[labelIndex] = 1.0;
			this.labels = new BasicVector(labelsArray);
		}
		public boolean checkGuess(Vector guess) {
			//Checks if the guess vector matches the label of this sample
			//Returns true iff the highest guessed value is the label of this sample
			double max = guess.max();
			int guessLabelIndex = -1;

			for(int i = 0; i < guess.length(); i++) {
				if(guess.get(i) == max) {
					guessLabelIndex = i;
					break;
				}
			}

			return guessLabelIndex == labelIndex;
		}
	}

	private Sample[] getSamples(String filename, int numRows) {
		//Load samples from CSV file
    	Sample[] s = new Sample[numRows];
        try {
			CSVReader reader = new CSVReader(new FileReader(filename));
		    String[] nextLine;
		    for (int row  = 0; row < numRows; row ++) {
		    	nextLine = reader.readNext();
			    double [] pixels = new double[28*28];
			    for(int i = 0; i < 28 * 28; i++) {
			    	pixels[i] = Integer.parseInt(nextLine[i + 1]) / 255.0;
			    }
			    int labelIndex = Integer.parseInt(nextLine[0]);
			    s[row] = new Sample(pixels, labelIndex);
		    }
		    reader.close();
		    return s;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	public Sample getTraining() {
		return training[rand.nextInt(60000)];
	}

	public Sample getTest() {
		return test[rand.nextInt(10000)];
	}

	public void initialize() {
		if(training == null) {
			training = getSamples("resources/mnist_train.csv", 60000);
		}
		if(test == null) {
			test = getSamples("resources/mnist_test.csv", 10000);
		}		
	}
}
