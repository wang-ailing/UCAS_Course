import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;


import java.io.IOException;
import java.lang.reflect.Array;
import java.math.BigDecimal;

import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.MasterNotRunningException;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.ZooKeeperConnectionException;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;

import org.apache.log4j.*;

import java.util.*;

/**
 * Hw1Grp2 Class
 *
 * This Java program processes an HDFS file by performing a "group by" operation on a specified column
 * and applying aggregation functions such as count, average, and max. The results are then stored in an HBase table.
 * 
 * Use BigDecimal to avoid precision errors when performing arithmetic operations.
 * 
 * Compile:
 * javac /home/bdms/homework/hw1/example/2_202428015059018_hw1.java
 * 
 * Usage:
 * java Hw1Grp2 R=<file> groupby:R<column> 'res:count,avg(Rx),max(Ry)'
 *
 * Example:
 * java Hw1Grp2 R=/myTest/region.tbl groupby:R0 'res:count,avg(R0),max(R0),count'
 * 
 * @author WangAiLing
 */

class  Hw1Grp2 {
    private static final String TABLE_NAME = "Result";
    private static final String COLUMN_FAMILY = "res";

    /**
     * Main method to process input arguments and execute the workflow.
     * Validates input format, reads and processes HDFS data, and stores results in HBase.
     *
     * @param args - Command-line input arguments
     * @throws Exception - If any error occurs during execution
     */
    public static void main(String[] args) throws Exception{
        if (args.length == 3) {
            if (args[0].matches("^R=(.+)$") == false ){
                System.out.println("Wrong:::  R=<file>");
                printInputFormat();
            }
            if (args[1].matches("^groupby:R(\\d+)$") == false ){
                System.out.println("Wrong:::  groupby:R<column>");
                printInputFormat();
            }
            String countAvgMaxRegex = "(count|avg\\(R\\d+\\)|max\\(R\\d+\\))";
            if (args[2].matches("^res:"+"("+countAvgMaxRegex+",)*"+countAvgMaxRegex+"$") == false){
                System.out.println("Wrong::: 'res:count,avg(R3),max(R4)'");
                printInputFormat();
            }
		}
        else printInputFormat();

        String filePath = args[0].substring("R=".length());
        Integer column = Integer.parseInt(args[1].substring("groupby:R".length()));
        // System.out.println(column);
        String[] operations = args[2].substring("res:".length()).split(","); 
        // for (String operation : operations)
        //     System.out.println(operation);

        Map<String, BigDecimal[]> result = readAndProcessHDFS(filePath, column, operations);
        
        System.out.print("Key\t");
        for (String operation : operations)
            System.out.print(operation + "\t");
        System.out.println();
        for (String key : result.keySet()) {
            System.out.print(key + "\t");
            for (BigDecimal bd : result.get(key)) {
                System.out.print(bd + "\t");
            }
            System.out.println();
        }

        createAndOutputHBaseTable(result, operations);


    }

    /**
     * This method reads a HDFS file and performs a group-by operation on a specified column.
     * @param file - HDFS file path
     * @param column - column to group by
     * @param operations - operations to perform on the grouped data
     * @return Map - key-value pairs of the grouped data
     * @throws IOException
     * @throws URISyntaxException
     */

    private static Map<String, BigDecimal[]> readAndProcessHDFS(String file, Integer column, String[] operations)  throws IOException, URISyntaxException{
        Configuration conf = new Configuration();

        FileSystem fs = FileSystem.get(URI.create(file), conf);
        Path path = new Path(file);
        FSDataInputStream in_stream = fs.open(path);

        BufferedReader in = new BufferedReader(new InputStreamReader(in_stream));

        Map<String, BigDecimal[]> result = processHDFS(in, column, operations);

        in.close();
        fs.close();
        return result;
    }


    /**
     * This method processes the data read from the HDFS file. 
     * Time complexity: O(n*opr) n is the number of lines in the HDFS file. opr is the number of operations.
     * This method could process redundant operations like "count,count,count", but it is not a primary problem.
     * @param in - BufferedReader object to read the HDFS file
     * @param column - column to group by
     * @param operations - operations to perform on the grouped data
     * @return Map - key-value pairs of the grouped data
     * @throws IOException
     */

    static Map<String, BigDecimal[]> processHDFS(BufferedReader in, Integer column, String[] operations) throws IOException {

        Set<Integer> avgColumnSet = new HashSet<Integer>();
        Set<Integer> maxColumnSet = new HashSet<Integer>();
        for (String operation : operations) {
            if (operation.matches("avg(.+)")) {
                avgColumnSet.add(Integer.parseInt(operation.substring("avg(R".length(), operation.length()-1)));
            }
            else if (operation.matches("max(.+)")) {
                maxColumnSet.add(Integer.parseInt(operation.substring("max(R".length(), operation.length()-1)));
            }
        }
        int[] avgColumns = avgColumnSet.stream().mapToInt(i->i).toArray();
        int[] maxColumns = maxColumnSet.stream().mapToInt(i->i).toArray();

        Map<Integer, Integer> avgColumnMap = new HashMap<Integer, Integer>();
        for (int i=0;i<avgColumns.length;i++) {
            avgColumnMap.put(avgColumns[i], i);
        }
        Map<Integer, Integer> maxColumnMap = new HashMap<Integer, Integer>();
        for (int i=0;i<maxColumns.length;i++) {
            maxColumnMap.put(maxColumns[i], i);
        }

        Map<String, Integer> countMap = new HashMap<String, Integer>();
        Map<String, BigDecimal[]> sumMap = new HashMap<String,BigDecimal[]>();
        Map<String, BigDecimal[]> maxMap = new HashMap<String,BigDecimal[]>();
        
        String lineString;
        while ((lineString=in.readLine())!=null) {
            // System.out.println(lineString);
            
            String[] line = lineString.split("\\|");
            // for (String s : line)
            //     System.out.print(s + " ");
            String key = line[column];
            countMap.merge(key, 1, Integer::sum);    // count++


            BigDecimal[] sumArray = sumMap.get(key);
            if (sumArray == null)
                sumArray = new BigDecimal[avgColumns.length];
            for (int i=0;i<avgColumns.length;i++) {
                int col = avgColumns[i];
                if (col >= line.length){
                    System.out.println("Wrong:::  avg(R"+col+")   More than the number of columns in the input file");
                    printInputFormat();
                }
                String value = line[col];
                if (sumArray[i] == null)
                    sumArray[i] = new BigDecimal(value);
                else 
                    sumArray[i] = sumArray[i].add(new BigDecimal(value));
            }
            sumMap.put(key, sumArray);

            BigDecimal[] maxArray = maxMap.get(key);
            if (maxArray == null)
                maxArray = new BigDecimal[maxColumns.length];
            for (int i=0;i<maxColumns.length;i++) {
                int col = maxColumns[i];
                if (col >= line.length){
                    System.out.println("Wrong:::  max(R"+col+")   More than the number of columns in the input file");
                    printInputFormat();
                }
                String value = line[col];

                if (maxArray[i] == null)
                    maxArray[i] = new BigDecimal(value);
                else 
                    maxArray[i] = maxArray[i].max(new BigDecimal(value));
            }
            maxMap.put(key, maxArray);
        }


        Map<String, BigDecimal[]> result = new HashMap<String, BigDecimal[]>();
        for (String key : countMap.keySet()) {
            BigDecimal[] resultArray = new BigDecimal[operations.length];
            // Arrays.fill(resultArray, new BigDecimal(0));
            result.put(key, resultArray);
        }

        for (int i=0;i<operations.length;i++){
            String operation = operations[i];
            if (operation.matches("count")) {
                for (String key : countMap.keySet()) {
                    Integer countMapInteger = countMap.get(key);
                    result.get(key)[i] = new BigDecimal(countMapInteger);
                }
            }
            else if (operation.matches("avg(.+)")) {
                int col = Integer.parseInt(operation.substring("avg(R".length(), operation.length()-1));
                int index = avgColumnMap.get(col);
                for (String key : countMap.keySet()) {
                    BigDecimal avg = sumMap.get(key)[index].divide(new BigDecimal(countMap.get(key)), 2, BigDecimal.ROUND_HALF_UP);
                    result.get(key)[i] = avg;
                }
            }
            else if (operation.matches("max(.+)")) {
                int col = Integer.parseInt(operation.substring("max(R".length(), operation.length()-1));
                int index = maxColumnMap.get(col);
                for (String key : maxMap.keySet()) {
                    result.get(key)[i] = maxMap.get(key)[index];
                }
            }
        }


        return result;
        // return null;
    }

    /**
     * This method creates a HBase table and outputs the grouped data to it.
     * @param result
     * @param operations
     * @throws MasterNotRunningException
     * @throws ZooKeeperConnectionException
     * @throws IOException
     */

    static void createAndOutputHBaseTable(Map<String, BigDecimal[]> result, String[] operations) throws MasterNotRunningException, ZooKeeperConnectionException, IOException  {
        Logger.getRootLogger().setLevel(Level.WARN);

        // System.out.println("Creating HBase table ....");
        // create table descriptor
        HTableDescriptor tableDescriptor = new HTableDescriptor(TableName.valueOf(TABLE_NAME));
        tableDescriptor.addFamily(new HColumnDescriptor(COLUMN_FAMILY));
        // System.out.println("tableDescriptor: " + tableDescriptor);
        // configure HBase
        Configuration configuration = HBaseConfiguration.create();
        HBaseAdmin hAdmin = new HBaseAdmin(configuration);
    
        if (hAdmin.tableExists(TABLE_NAME)) {
            System.out.println("Table " + TABLE_NAME + " already exists");
            hAdmin.disableTable(TABLE_NAME);
            hAdmin.deleteTable(TABLE_NAME);
            System.out.println("Table " + TABLE_NAME + " deleted successfully");
        }

        hAdmin.createTable(tableDescriptor);
        System.out.println("Table "+ TABLE_NAME + " created successfully");
        hAdmin.close();
    
        // put "Result", myKey, "res:count","3"
    
        HTable table = new HTable(configuration,TABLE_NAME);


        for (String key : result.keySet()) {
            BigDecimal[] values = result.get(key);
            for (int i=0;i<values.length;i++){
                String operation = operations[i];
                Put put = new Put(key.getBytes());
                put.add(COLUMN_FAMILY.getBytes(), operation.getBytes(), values[i].toString().getBytes());
                table.put(put);
            }
        }

        table.close();
        System.out.println("Table " + TABLE_NAME + " closed successfully");
    }


    /**
     * This method prints the input format for the program.
     */
    static void printInputFormat(){
        // System.out.println("Wrong Arguments!!!");
        System.out.println("Usage: java Hw1Grp2 R=<file> groupby:R2 'res:count,avg(R3),max(R4)'");
        // System.out.println("Usage: HDFSTest <hdfs-file-path>");
        System.exit(1);
    }
}