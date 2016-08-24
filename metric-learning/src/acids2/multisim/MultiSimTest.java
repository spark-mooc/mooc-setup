package acids2.multisim;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import acids2.Resource;
import acids2.output.Fscore;
import au.com.bytecode.opencsv.CSVReader;

public class MultiSimTest {

	private static ArrayList<Resource> sources = new ArrayList<Resource>();
	private static ArrayList<Resource> targets = new ArrayList<Resource>();
	
    private static ArrayList<String> oraclesAnswers = new ArrayList<String>();
	private static ArrayList<String> ignoredList = new ArrayList<String>();
    
    static {
    	ignoredList.add("id");
    	ignoredList.add("venue");
//    	ignoredList.add("manufacturer");
//    	ignoredList.add("description");
//    	ignoredList.add("price");
    }

    /**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		
		String datasetPath = args[0];
		int K = Integer.parseInt(args[1]);
		
		String sourcePath = "data/" + datasetPath + "/sources.csv";
		String targetPath = "data/" + datasetPath + "/targets.csv";
		loadKnowledgeBases(sourcePath, targetPath);
		
		String mappingPath = "data/" + datasetPath + "/mapping.csv";
		loadMappings(mappingPath);
		
		System.out.println("Starting MultiSim on: dataset = "+datasetPath+"\t|training set| = "+K);
		int N = 1;
		ArrayList<Fscore> fsl = new ArrayList<Fscore>();
		MultiSimOracle oracle = new MultiSimOracle(oraclesAnswers);
		for(int i=0; i<N; i++) {
			MultiSimSetting ma = new MultiSimSetting(sources, targets, oracle, K, datasetPath);
			ma.run();
			
			fsl.add(ma.getFs());
		}
//		try {
//			FscoreWriter.write(datasetPath + "_" + tfidf + "_" + H, fsl);
//		} catch (NullPointerException e) {
//			System.out.println("No f-score to print out.");
//		}
	}

    private static void loadKnowledgeBases(String sourcePath, String targetPath) throws IOException {
    	loadKnowledgeBases(sourcePath, targetPath, 0, Integer.MAX_VALUE);
    }

    private static void loadKnowledgeBases(String sourcePath, String targetPath, int startOffset, int endOffset) throws IOException {
    	
        CSVReader reader = new CSVReader(new FileReader(sourcePath));
        String [] titles = reader.readNext(); // gets the column titles
        for(int i=0; i<startOffset; i++) // skips start offset
        	reader.readNext();
        String [] nextLine;
        int count = 0;
        while ((nextLine = reader.readNext()) != null) {
            Resource r = new Resource(nextLine[0]);
            for(int i=0; i<nextLine.length; i++) {
                if(!ignoredList.contains( titles[i].toLowerCase() )) {
                    if(nextLine[i] != null)
                        r.setPropertyValue(titles[i], nextLine[i]);
                    else
                        r.setPropertyValue(titles[i], "");
                }
            }
            sources.add(r);
            if(++count >= endOffset)
            	break;
        }
        
        reader = new CSVReader(new FileReader(targetPath));
        titles = reader.readNext(); // gets the column titles
        for(int i=0; i<startOffset; i++) // skips offset
        	reader.readNext();
        count = 0;
        while ((nextLine = reader.readNext()) != null) {
            Resource r = new Resource(nextLine[0]);
            for(int i=0; i<nextLine.length; i++)
                if(!ignoredList.contains( titles[i].toLowerCase() )) {
                    if(nextLine[i] != null)
                        r.setPropertyValue(titles[i], nextLine[i]);
                    else
                        r.setPropertyValue(titles[i], "");
                }
            targets.add(r);
            if(++count >= endOffset)
            	break;
        }
        
        reader.close();
    }

	private static void loadMappings(String mappingPath) throws IOException {
        CSVReader reader = new CSVReader(new FileReader(mappingPath));
        reader.readNext(); // skips the column titles
        String [] nextLine;
        while ((nextLine = reader.readNext()) != null) {
            oraclesAnswers.add(nextLine[0] + "#" + nextLine[1]);
        }
        reader.close();
    }

}
