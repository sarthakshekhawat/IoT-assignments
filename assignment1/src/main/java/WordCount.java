import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.*;
import java.util.*;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

public class WordCount {
    public static class MyMapper extends
            Mapper<LongWritable, Text, Text, IntWritable> {
        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);
            while (tokenizer.hasMoreTokens()) {
                word.set(tokenizer.nextToken());
                context.write(word, new IntWritable(1));
            }
        }
    }

    public static class MyReducer extends
            Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            context.write(key, new IntWritable(sum));
        }
    }

    public static class RelabeledMapper extends
            Mapper<LongWritable, Text, IntWritable, Text> {
        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);
            String newLine = "";
            int numOfEntriesInColumn = 0;
            while (tokenizer.hasMoreTokens()) {
                numOfEntriesInColumn++;
                word.set(getRelabeledToken(tokenizer.nextToken()));
                newLine += word + "\t";
            }
            context.write(new IntWritable(-numOfEntriesInColumn), new Text(newLine));
        }
    }

    public static class RelabeledReducer extends
            Reducer<IntWritable, Text, IntWritable, Text> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            FileWriter fr = new FileWriter("output/relabeledMatrix", true);
            for (Text val: values) {
                String newVal = getSortedLine(val.toString());
                fr.write(newVal + "\n");
            }
            fr.close();
        }
    }

    public static class GroupMapper extends
            Mapper<LongWritable, Text, IntWritable, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);
            HashMap<Integer, Boolean> groupFoundInPresentLine = new HashMap<Integer, Boolean>();
            while (tokenizer.hasMoreTokens()) {
                int groupLabel = -1;
                try {
                    groupLabel = getGroupLabel(Integer.valueOf(tokenizer.nextToken()));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                if (groupFoundInPresentLine.get(groupLabel) == null) {
                    groupFoundInPresentLine.put(groupLabel, true);
                    context.write(new IntWritable(groupLabel), new Text(line));
                }
            }
        }
    }

    public static class GroupReducer extends
            Reducer<IntWritable, Text, IntWritable, Text> {

        private MultipleOutputs multipleOutputs;

        protected void setup(Context context) throws IOException, InterruptedException {
            multipleOutputs  = new MultipleOutputs(context);
        }

        public void reduce(IntWritable key, Iterable<Text> values,
                           Context context) throws IOException, InterruptedException {
            String output = "";
            for (Text val : values) {
                output += val.toString() + "\n";
            }

            multipleOutputs.write((Object) null, new Text(output), "Group_" + key.toString());
        }

        protected void cleanup(Context context)
                throws IOException, InterruptedException {
            multipleOutputs.close();
        }
    }

    public static int getGroupLabel (int index) throws Exception {
        int groupLabel;
        File file = new File("output/relabelIndexGroups");
        BufferedReader br = new BufferedReader(new FileReader(file));

        String st;
        while ((st = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(st);
            int relabeledIndex = Integer.valueOf(tokenizer.nextToken());
            if (relabeledIndex == index)
                return Integer.valueOf(tokenizer.nextToken());
        }
        return -1;
    }

    public static String getSortedLine(String val) {
        StringTokenizer tokenizer = new StringTokenizer(val);
        List<Integer> IntTokens = new ArrayList<Integer>();
        while(tokenizer.hasMoreTokens()) {
            IntTokens.add(Integer.valueOf(tokenizer.nextToken()));
        }
        Collections.sort(IntTokens, Collections.reverseOrder());
        String newVal = "";
        for(int index = 0; index < IntTokens.size(); index++) {
            newVal += IntTokens.get(index).toString();
            if(index != IntTokens.size()-1)
                newVal += "\t";
        }
        return newVal;
    }

    public static String getRelabeledToken(String originalVal) throws IOException {
        File file = new File("output/relabel");
        BufferedReader br = new BufferedReader(new FileReader(file));

        String st;
        while ((st = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(st);
            String newVal = tokenizer.nextToken();
            if (newVal.equals(originalVal))
                return tokenizer.nextToken();
        }
        return "Error";
    }

    public static void groupRelabels(int threshold) throws IOException {
        // hashMap where key is relabel of element and value is original element.
        HashMap<Integer, Integer> reverseRelabels = new HashMap<Integer, Integer>();
        File file = new File("output/relabel");
        BufferedReader br = new BufferedReader(new FileReader(file));

        String st;

        while ((st = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(st);
            int x = Integer.valueOf(tokenizer.nextToken()), y = Integer.valueOf(tokenizer.nextToken());
            reverseRelabels.put(y, x);
        }
        br.close();

        HashMap<Integer, Integer> frequency = new HashMap<Integer, Integer>();
        file = new File("output/part-r-00000");
        br = new BufferedReader(new FileReader(file));

        while ((st = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(st);
            int x = Integer.valueOf(tokenizer.nextToken()), y = Integer.valueOf(tokenizer.nextToken());
            frequency.put(x, y);
        }
        br.close();

        Map<Integer, Integer> relabeledIndexFrequency = new TreeMap<Integer, Integer>();
        ArrayList<Integer> keys = new ArrayList<Integer>(reverseRelabels.keySet());
        for (int i = 0; i < keys.size(); i++) {
            int x = keys.get(i);
            int y1 = reverseRelabels.get(x);
            int y2 = frequency.get(y1);
            relabeledIndexFrequency.put(x, y2);
        }

        writeToFileKeyValue("output/relabelIndexFrequency", relabeledIndexFrequency);

        HashMap<Integer, Integer> relabeledIndexGroups;
        relabeledIndexGroups = getRelabeledIndexGroups(relabeledIndexFrequency, threshold);

        /*  This file consists of relabeled indexes and corresponding group numbers.
            The same is used while making group files.
         */
        writeToFileKeyValue("output/relabelIndexGroups", relabeledIndexGroups);
    }

    public static void writeToFileKeyValue(String fileName, Map<Integer, Integer> map) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));

        for (HashMap.Entry<Integer, Integer> entry : map.entrySet()) {
            writer.write(entry.getKey() + " " + entry.getValue() + "\n");
        }

        writer.close();
    }

    public static HashMap<Integer, Integer> getRelabeledIndexGroups(Map<Integer, Integer> map, int threshold) {
        HashMap<Integer, Integer> groupingMap = new HashMap<Integer, Integer>();

        ArrayList<Integer> keys = new ArrayList<Integer>(map.keySet());
        int sum = 0, group = 0;
        for (int i = keys.size()-1; i >= 0; i--) {
            sum += map.get(keys.get(i));
            if (sum >= threshold) {
                if (sum != map.get(keys.get(i))) {
                    group++;
                    i++;
                }

                sum = 0;
                continue;
            }

            groupingMap.put(keys.get(i), group);
        }
        return groupingMap;
    }

    public static <K, V extends Comparable<V>> Map<K, V>
    sortByValues(final Map<K, V> map) {
        Comparator<K> valueComparator =
                new Comparator<K>() {
                    public int compare(K k1, K k2) {
                        int compare =
                                map.get(k2).compareTo(map.get(k1));
                        if (compare == 0)
                            return 1;
                        else
                            return compare;
                    }
                };

        Map<K, V> sortedByValues =
                new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll(map);
        return sortedByValues;
    }

    public static void relabel() throws IOException {
        File file = new File("output/part-r-00000");
        BufferedReader br = new BufferedReader(new FileReader(file));
        TreeMap<Integer, Integer> tm = new TreeMap<Integer, Integer>();

        String st;
        while ((st = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(st);
            int x = Integer.valueOf(tokenizer.nextToken()), y = Integer.valueOf(tokenizer.nextToken());
            tm.put(x, y);
        }

        Map sortedMap = sortByValues(tm);
        Set set = sortedMap.entrySet();
        Iterator i = set.iterator();

        BufferedWriter writer = new BufferedWriter(new FileWriter("output/relabel"));

        Integer index = 1;
        while(i.hasNext()) {
            Map.Entry me = (Map.Entry)i.next();
            Integer key = (Integer) me.getKey();
            writer.write(key + " " + index + "\n");
            index++;
        }

        writer.close();
    }

    public static void groupRelabelsPrint(String[] args) throws Exception {
        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(WordCount.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        job.setMapperClass(GroupMapper.class);
        job.setReducerClass(GroupReducer.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job, new Path("output/relabeledMatrix"));
        FileOutputFormat.setOutputPath(job, new Path("output2"));
        boolean status = job.waitForCompletion(true);
        if (status) {
            System.exit(0);
        } else {
            System.exit(1);
        }
    }

    public static void relabeledMatrix(String[] args) throws Exception {
        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(WordCount.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        job.setMapperClass(RelabeledMapper.class);
        job.setReducerClass(RelabeledReducer.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("output1"));
        boolean status = job.waitForCompletion(true);
        if (status) {
            int threshold = 5;
            groupRelabels(threshold);
            groupRelabelsPrint(args);
            System.exit(0);
        } else {
            System.exit(1);
        }
    }

    public static void main(String[] args) throws Exception {
        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(WordCount.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("output"));
        boolean status = job.waitForCompletion(true);
        if (status) {
            relabel();
            relabeledMatrix(args);
            System.exit(0);
        } else {
            System.exit(1);
        }
    }
}