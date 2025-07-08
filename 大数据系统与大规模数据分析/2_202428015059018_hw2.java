/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified by Shimin Chen to demonstrate functionality for Homework 2
// April-May 2015

import java.io.IOException;
import java.math.BigDecimal;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import javax.naming.Context;

public class Hw2Part1 {

    // This is the Mapper class
    // reference: http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/mapreduce/Mapper.html
    //
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, DoubleWritable>{

        private final static DoubleWritable duration = new DoubleWritable();
        private Text word = new Text();

        // This method is called for each line of input
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            int tokenCount = itr.countTokens();
            System.out.println("Token count: " + tokenCount);
            if (tokenCount != 3) return;
            String source = itr.nextToken();
            String destination = itr.nextToken();
            double amount = Double.parseDouble(itr.nextToken());
            word.set(source + " " + destination);
            System.out.println("Source: " + source + " Destination: " + destination);
            duration.set(amount);
            context.write(word, duration);
//            String duration = itr.nextToken();
//            while (itr.hasMoreTokens()) {
//                String nextToken = itr.nextToken();
//                System.out.println(nextToken);
//                word.set(word);
//                context.write(word, one);
//            }
        }
    }


    // This is the Reducer class
    // reference http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/mapreduce/Reducer.html
    //
    // We want to control the output format to look at the following:
    //
    // count of word = count
    //
    public static class IntSumReducer
            extends Reducer<Text, DoubleWritable,Text,Text> {
        Text result = new Text();

        // This method is called for each key-value pair
        public void reduce(Text key, Iterable<DoubleWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int cnt = 0;
            BigDecimal sum = new BigDecimal(0);
            for (DoubleWritable val : values) {
                cnt++;
                sum = sum.add(new BigDecimal(val.get()));
            }
            BigDecimal average = sum.divide(new BigDecimal(cnt), 3, BigDecimal.ROUND_HALF_UP);
            result.set(String.valueOf(cnt) + " " + average.toString());
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length < 2) {
            System.err.println("Usage: Hw2Part1 <in> [<in>...] <out>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "word count"); // job name Not important

        job.setJarByClass(Hw2Part1.class);

        job.setMapperClass(TokenizerMapper.class);
//        job.setCombinerClass(IntSumCombiner.class);
        job.setReducerClass(IntSumReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // add the input paths as given by command line
        for (int i = 0; i < otherArgs.length - 1; ++i) {
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }

        // add the output path as given by the command line
        FileOutputFormat.setOutputPath(job,
                new Path(otherArgs[otherArgs.length - 1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
