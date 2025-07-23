package cli.clt.bedpe;

import cli.clt.CommandLineParser;
import cli.utils.apa.MultiAPAManager;
import cli.utils.data.SparseContactMatrixWithMasking;

import io.jhdf.HdfFile;
import io.jhdf.WritableHdfFile;
import io.jhdf.api.WritableGroup;

import javastraw.feature2D.Feature2D;
import javastraw.feature2D.Feature2DList;
import javastraw.feature2D.Feature2DParser;
import javastraw.reader.Dataset;
import javastraw.reader.basics.Chromosome;
import javastraw.reader.basics.ChromosomeHandler;
import javastraw.reader.mzd.Matrix;
import javastraw.reader.mzd.MatrixZoomData;
import javastraw.reader.norm.NormalizationPicker;
import javastraw.reader.type.HiCZoom;
import javastraw.reader.type.NormalizationType;
import javastraw.tools.HiCFileTools;
import javastraw.tools.ParallelizationTools;

import java.io.File;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class Grind {

    public static String usage = "grind [-k NORM] [-r resolution] [--window half-width] [--npy] <hic file> <bedpe> <directory> <stem>\n" +
            "\t\tsplit up the bedpe into multiple lists; number < 2 splits by chromosome";
    private final boolean useObservedOverExpected = false;
    private final boolean useNpy;
    private final Dataset ds;
    private final File outputDirectory;
    private final HiCZoom zoom;
    private final Feature2DList loopList;
    private final ChromosomeHandler handler;
    private int matrixHalfWidth = 10;
    private Integer resolution = 1000;
    private NormalizationType norm;
    private String stemName;

    // Lists to store loop information
    private final List<String> chr1List = new ArrayList<>();
    private final List<Long> start1List = new ArrayList<>();
    private final List<Long> end1List = new ArrayList<>();
    private final List<String> chr2List = new ArrayList<>();
    private final List<Long> start2List = new ArrayList<>();
    private final List<Long> end2List = new ArrayList<>();
    private final List<Integer> chunkIndexList = new ArrayList<>();

    // Class to store loop data and its matrix
    private static class LoopData {
        String chr1;
        long start1;
        long end1;
        String chr2;
        long start2;
        long end2;
        float[][] matrix;

        LoopData(String chr1, long start1, long end1, String chr2, long start2, long end2, float[][] matrix) {
            this.chr1 = chr1;
            this.start1 = start1;
            this.end1 = end1;
            this.chr2 = chr2;
            this.start2 = start2;
            this.end2 = end2;
            this.matrix = matrix;
        }
    }

    // Processing mode based on dataset size
    private enum ProcessingMode {
        SMALL,   // <1M loops, full in-memory
        MEDIUM,  // 1-8M loops, bounded queues
        LARGE    // >8M loops, streaming pipeline
    }

    // Queue to store loop data
    private final ConcurrentLinkedQueue<LoopData> loopDataQueue = new ConcurrentLinkedQueue<>();

    // Lock object for synchronizing access to loop info lists
    private final Object loopInfoLock = new Object();

    // Thread pool and processing configuration
    private final int numThreads;
    private final ProcessingMode processingMode;
    private final int maxLoopsInMemory;
    
    // Thread-local matrix pool for reuse
    private final ThreadLocal<float[][]> matrixPool = ThreadLocal.withInitial(() -> {
        int matrixWidth = 2 * matrixHalfWidth + 1;
        return new float[matrixWidth][matrixWidth];
    });

    public Grind(String[] args, CommandLineParser parser) {
        if (args.length != 5) {
            printUsageAndExit();
        }

        resolution = parser.getResolutionOption(5000);
        boolean useBI = resolution >= 50;

        ds = HiCFileTools.extractDatasetForCLT(args[1], true, false, useBI);
        outputDirectory = HiCFileTools.createValidDirectory(args[3]);

        stemName = sanitizeStemName(args[4]);

        String possibleNorm = parser.getNormalizationStringOption();

        try {
            norm = ds.getNormalizationHandler().getNormTypeFromString(possibleNorm);
        } catch (Exception e) {
            norm = NormalizationPicker.getFirstValidNormInThisOrder(ds, new String[]{possibleNorm, "SCALE", "KR", "NONE"});
        }

        System.out.println("Using normalization: " + norm.getLabel());

        matrixHalfWidth = parser.getWindowSizeOption(10);

        useNpy = parser.getNpyOption();

        zoom = new HiCZoom(resolution);
        handler = ds.getChromosomeHandler();

        loopList = Feature2DParser.loadFeatures(args[2], handler, false, null, false);
        if (loopList.getNumTotalFeatures() < 1) {
            System.err.println("Loop list is empty or incorrect path provided.");
            System.exit(3);
        }
        // Calculate processing configuration
        numThreads = Runtime.getRuntime().availableProcessors();
        long availableMemory = Runtime.getRuntime().maxMemory();
        int totalLoops = loopList.getNumTotalFeatures();
        int matrixWidth = 2 * matrixHalfWidth + 1;
        long memoryPerLoop = 200 + (matrixWidth * matrixWidth * 4L); // ~12KB for 55x55
        maxLoopsInMemory = (int) Math.min(totalLoops, (availableMemory * 0.8) / memoryPerLoop);
        
        processingMode = determineProcessingMode(totalLoops, maxLoopsInMemory);
        
        System.out.println("Using stem name: " + stemName);
        System.out.println("Total loops: " + totalLoops);
        System.out.println("Matrix size: " + matrixWidth + "x" + matrixWidth);
        System.out.println("Memory per loop: " + memoryPerLoop + " bytes");
        System.out.println("Max loops in memory: " + maxLoopsInMemory);
        System.out.println("Processing mode: " + processingMode);
        System.out.println("Using " + numThreads + " threads");
    }

    public void run() {
        if (useNpy) {
            System.out.println("Using Numpy file type");
            LoopDumper.dump(ds, loopList, outputDirectory, handler, norm,
                    useObservedOverExpected, resolution, matrixHalfWidth);
        } else {
            System.out.println("Using HDF5 file type");
            String resolutionGroupName = String.valueOf(resolution);

            File hdf5File = new File(outputDirectory, stemName + "_loops_at_" + resolutionGroupName + "_bp.hdf5");

            try (WritableHdfFile writableHdfFile = HdfFile.write(Paths.get(hdf5File.getAbsolutePath()))) {

                WritableGroup resolutionGroup = writableHdfFile.putGroup(resolutionGroupName);
                WritableGroup chunksGroup = resolutionGroup.putGroup("chunks");
                WritableGroup loopInfoGroup = resolutionGroup.putGroup("loop_info");

                // Process all chromosomes using adaptive strategy
                switch (processingMode) {
                    case SMALL:
                        processAllChromosomesInMemory();
                        break;
                    case MEDIUM:
                        processAllChromosomesBounded();
                        break;
                    case LARGE:
                        processAllChromosomesStreaming(chunksGroup);
                        break;
                }

                // Process the collected loop data and write to HDF5 (for SMALL/MEDIUM modes)
                if (processingMode != ProcessingMode.LARGE) {
                    writeLoopDataToHDF5(chunksGroup);
                }

                // Write loop info to HDF5
                loopInfoGroup.putDataset("chr1", chr1List.toArray(new String[0]));
                loopInfoGroup.putDataset("start1", start1List.stream().mapToLong(Long::longValue).toArray());
                loopInfoGroup.putDataset("end1", end1List.stream().mapToLong(Long::longValue).toArray());
                loopInfoGroup.putDataset("chr2", chr2List.toArray(new String[0]));
                loopInfoGroup.putDataset("start2", start2List.stream().mapToLong(Long::longValue).toArray());
                loopInfoGroup.putDataset("end2", end2List.stream().mapToLong(Long::longValue).toArray());
                loopInfoGroup.putDataset("chunk_index", chunkIndexList.stream().mapToInt(Integer::intValue).toArray());

                System.out.println("All loops have been successfully written to " + hdf5File.getAbsolutePath());

            } catch (Exception e) {
                System.err.println("Error writing to HDF5 file.");
                e.printStackTrace();
                System.exit(4);
            }
        }
    }

    private void processAllChromosomes() {
        int matrixWidth = 2 * matrixHalfWidth + 1;

        loopList.processLists((chr, feature2DList) -> {
            System.out.println("Currently on: " + chr);

            Chromosome chrom = handler.getChromosomeFromName(feature2DList.get(0).getChr1());

            Matrix matrix = ds.getMatrix(chrom, chrom);
            if (matrix == null) return;

            HiCZoom zoom = ds.getZoomForBPResolution(resolution);
            final MatrixZoomData zd = matrix.getZoomData(zoom);

            if (zd == null) return;

            try {
                SparseContactMatrixWithMasking scm = new SparseContactMatrixWithMasking(zd,
                        feature2DList, resolution, matrixHalfWidth, matrixWidth, norm, true);

                AtomicInteger index = new AtomicInteger(0);

                ParallelizationTools.launchParallelizedCode(numThreads, () -> {
                    int currIndex = index.getAndIncrement();

                    while (currIndex < feature2DList.size()) {
                        Feature2D loop = feature2DList.get(currIndex);
                        float[][] output = new float[matrixWidth][matrixWidth];
                        MultiAPAManager.addToMatrix(output, scm, loop, matrixHalfWidth, resolution, matrixWidth);

                        // Add loop data to queue
                        loopDataQueue.add(new LoopData(
                                loop.getChr1(),
                                loop.getStart1(),
                                loop.getEnd1(),
                                loop.getChr2(),
                                loop.getStart2(),
                                loop.getEnd2(),
                                output
                        ));

                        currIndex = index.getAndIncrement();
                    }
                });

            } catch (Exception ex) {
                System.err.println("Error processing: " + chr);
                ex.printStackTrace();
            }
        });
    }

    private void writeLoopDataToHDF5(WritableGroup chunksGroup) {
        int matrixWidth = 2 * matrixHalfWidth + 1;
        AtomicInteger chunkIndex = new AtomicInteger(0);
        List<LoopData> currentChunk = new ArrayList<>(100);

        // Process all loops from the queue
        while (!loopDataQueue.isEmpty() || !currentChunk.isEmpty()) {
            // Fill current chunk up to 100 loops
            while (currentChunk.size() < 100 && !loopDataQueue.isEmpty()) {
                LoopData loopData = loopDataQueue.poll();
                if (loopData != null) {
                    currentChunk.add(loopData);
                }
            }

            // Write chunk if not empty
            if (!currentChunk.isEmpty()) {
                int chunkSize = currentChunk.size();
                float[][][] chunkArray = new float[chunkSize][matrixWidth][matrixWidth];

                // Create the chunk array
                for (int i = 0; i < chunkSize; i++) {
                    chunkArray[i] = currentChunk.get(i).matrix;
                }

                // Write the chunk to HDF5
                int currentChunkIndex = chunkIndex.getAndIncrement();
                String datasetName = "chunk_" + currentChunkIndex;
                chunksGroup.putDataset(datasetName, chunkArray);

                // Add loop info to lists
                synchronized (loopInfoLock) {
                    for (LoopData loopData : currentChunk) {
                        chr1List.add(loopData.chr1);
                        start1List.add(loopData.start1);
                        end1List.add(loopData.end1);
                        chr2List.add(loopData.chr2);
                        start2List.add(loopData.start2);
                        end2List.add(loopData.end2);
                        chunkIndexList.add(currentChunkIndex);
                    }
                }

                // Clear the current chunk
                currentChunk.clear();
            }
        }

        System.out.println("Wrote " + chunkIndex.get() + " chunks to HDF5 file");
    }

    private String sanitizeStemName(String stemName) {
        return stemName.replaceAll("[\\\\/:*?\"<>|]", "_");
    }

    private ProcessingMode determineProcessingMode(int totalLoops, int maxLoopsInMemory) {
        if (totalLoops <= maxLoopsInMemory && totalLoops < 1_000_000) {
            return ProcessingMode.SMALL;
        } else if (totalLoops <= maxLoopsInMemory && totalLoops < 8_000_000) {
            return ProcessingMode.MEDIUM;
        } else {
            return ProcessingMode.LARGE;
        }
    }

    // SMALL mode: Full in-memory processing with chromosome parallelization
    private void processAllChromosomesInMemory() {
        System.out.println("Using SMALL dataset processing mode (in-memory)");
        
        // Get all chromosomes and process them in parallel
        Map<String, List<Feature2D>> chromosomeLoops = new TreeMap<>();
        loopList.processLists((chr, feature2DList) -> {
            chromosomeLoops.put(chr, new ArrayList<>(feature2DList));
        });
        
        // Process chromosomes in parallel
        chromosomeLoops.entrySet().parallelStream().forEach(entry -> {
            processChromosomeOptimized(entry.getKey(), entry.getValue());
        });
    }

    // MEDIUM mode: Bounded processing with backpressure
    private void processAllChromosomesBounded() {
        System.out.println("Using MEDIUM dataset processing mode (bounded)");
        
        BlockingQueue<LoopData> boundedQueue = new ArrayBlockingQueue<>(maxLoopsInMemory);
        
        // Producer: process chromosomes
        CompletableFuture<Void> producer = CompletableFuture.runAsync(() -> {
            loopList.processLists((chr, feature2DList) -> {
                processChromosomeWithBoundedQueue(chr, feature2DList, boundedQueue);
            });
        });
        
        // Consumer: transfer to main queue
        CompletableFuture<Void> consumer = CompletableFuture.runAsync(() -> {
            try {
                LoopData data;
                while (!producer.isDone() || !boundedQueue.isEmpty()) {
                    data = boundedQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (data != null) {
                        loopDataQueue.add(data);
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        CompletableFuture.allOf(producer, consumer).join();
    }

    // LARGE mode: Streaming processing with immediate writing
    private void processAllChromosomesStreaming(WritableGroup chunksGroup) {
        System.out.println("Using LARGE dataset processing mode (streaming)");
        
        AtomicInteger globalChunkIndex = new AtomicInteger(0);
        AtomicInteger globalLoopIndex = new AtomicInteger(0);
        
        // Process chromosomes sequentially to maintain order
        loopList.processLists((chr, feature2DList) -> {
            System.out.println("Streaming processing: " + chr);
            processChromosomeStreaming(chr, feature2DList, chunksGroup, globalChunkIndex, globalLoopIndex);
        });
    }

    private void processChromosomeOptimized(String chr, List<Feature2D> feature2DList) {
        System.out.println("Processing (optimized): " + chr);
        
        if (feature2DList.isEmpty()) return;
        
        Chromosome chrom = handler.getChromosomeFromName(feature2DList.get(0).getChr1());
        Matrix matrix = ds.getMatrix(chrom, chrom);
        if (matrix == null) return;
        
        HiCZoom zoom = ds.getZoomForBPResolution(resolution);
        final MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) return;
        
        try {
            int matrixWidth = 2 * matrixHalfWidth + 1;
            SparseContactMatrixWithMasking scm = new SparseContactMatrixWithMasking(zd,
                    feature2DList, resolution, matrixHalfWidth, matrixWidth, norm, true);
            
            AtomicInteger index = new AtomicInteger(0);
            
            ParallelizationTools.launchParallelizedCode(numThreads, () -> {
                float[][] reusableMatrix = matrixPool.get();
                List<LoopData> localBatch = new ArrayList<>(1000);
                
                int currIndex = index.getAndIncrement();
                while (currIndex < feature2DList.size()) {
                    Feature2D loop = feature2DList.get(currIndex);
                    
                    // Clear and reuse matrix
                    clearMatrix(reusableMatrix);
                    MultiAPAManager.addToMatrix(reusableMatrix, scm, loop, matrixHalfWidth, resolution, matrixWidth);
                    
                    // Create copy for storage
                    float[][] matrixCopy = copyMatrix(reusableMatrix);
                    
                    localBatch.add(new LoopData(
                            loop.getChr1(),
                            loop.getStart1(),
                            loop.getEnd1(),
                            loop.getChr2(),
                            loop.getStart2(),
                            loop.getEnd2(),
                            matrixCopy
                    ));
                    
                    // Batch add to main queue
                    if (localBatch.size() >= 1000) {
                        loopDataQueue.addAll(localBatch);
                        localBatch.clear();
                    }
                    
                    currIndex = index.getAndIncrement();
                }
                
                // Add remaining items
                if (!localBatch.isEmpty()) {
                    loopDataQueue.addAll(localBatch);
                }
            });
            
        } catch (Exception ex) {
            System.err.println("Error processing: " + chr);
            ex.printStackTrace();
        }
    }

    private void processChromosomeWithBoundedQueue(String chr, List<Feature2D> feature2DList, BlockingQueue<LoopData> queue) {
        System.out.println("Processing (bounded): " + chr);
        
        if (feature2DList.isEmpty()) return;
        
        Chromosome chrom = handler.getChromosomeFromName(feature2DList.get(0).getChr1());
        Matrix matrix = ds.getMatrix(chrom, chrom);
        if (matrix == null) return;
        
        HiCZoom zoom = ds.getZoomForBPResolution(resolution);
        final MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) return;
        
        try {
            int matrixWidth = 2 * matrixHalfWidth + 1;
            SparseContactMatrixWithMasking scm = new SparseContactMatrixWithMasking(zd,
                    feature2DList, resolution, matrixHalfWidth, matrixWidth, norm, true);
            
            AtomicInteger index = new AtomicInteger(0);
            
            ParallelizationTools.launchParallelizedCode(numThreads, () -> {
                float[][] reusableMatrix = matrixPool.get();
                
                int currIndex = index.getAndIncrement();
                while (currIndex < feature2DList.size()) {
                    Feature2D loop = feature2DList.get(currIndex);
                    
                    clearMatrix(reusableMatrix);
                    MultiAPAManager.addToMatrix(reusableMatrix, scm, loop, matrixHalfWidth, resolution, matrixWidth);
                    
                    LoopData loopData = new LoopData(
                            loop.getChr1(),
                            loop.getStart1(),
                            loop.getEnd1(),
                            loop.getChr2(),
                            loop.getStart2(),
                            loop.getEnd2(),
                            copyMatrix(reusableMatrix)
                    );
                    
                    try {
                        queue.put(loopData); // Blocks if queue is full
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                    
                    currIndex = index.getAndIncrement();
                }
            });
            
        } catch (Exception ex) {
            System.err.println("Error processing: " + chr);
            ex.printStackTrace();
        }
    }

    private void processChromosomeStreaming(String chr, List<Feature2D> feature2DList, 
                                           WritableGroup chunksGroup, AtomicInteger globalChunkIndex, 
                                           AtomicInteger globalLoopIndex) {
        if (feature2DList.isEmpty()) return;
        
        Chromosome chrom = handler.getChromosomeFromName(feature2DList.get(0).getChr1());
        Matrix matrix = ds.getMatrix(chrom, chrom);
        if (matrix == null) return;
        
        HiCZoom zoom = ds.getZoomForBPResolution(resolution);
        final MatrixZoomData zd = matrix.getZoomData(zoom);
        if (zd == null) return;
        
        try {
            int matrixWidth = 2 * matrixHalfWidth + 1;
            SparseContactMatrixWithMasking scm = new SparseContactMatrixWithMasking(zd,
                    feature2DList, resolution, matrixHalfWidth, matrixWidth, norm, true);
            
            // Process in chunks of 100 loops directly to HDF5
            List<LoopData> currentChunk = new ArrayList<>(100);
            
            for (Feature2D loop : feature2DList) {
                float[][] loopMatrix = new float[matrixWidth][matrixWidth];
                MultiAPAManager.addToMatrix(loopMatrix, scm, loop, matrixHalfWidth, resolution, matrixWidth);
                
                currentChunk.add(new LoopData(
                        loop.getChr1(),
                        loop.getStart1(),
                        loop.getEnd1(),
                        loop.getChr2(),
                        loop.getStart2(),
                        loop.getEnd2(),
                        loopMatrix
                ));
                
                // Write chunk when full
                if (currentChunk.size() >= 100) {
                    writeChunkDirectly(currentChunk, chunksGroup, globalChunkIndex.getAndIncrement(), matrixWidth);
                    currentChunk.clear();
                }
            }
            
            // Write remaining loops
            if (!currentChunk.isEmpty()) {
                writeChunkDirectly(currentChunk, chunksGroup, globalChunkIndex.getAndIncrement(), matrixWidth);
            }
            
        } catch (Exception ex) {
            System.err.println("Error processing: " + chr);
            ex.printStackTrace();
        }
    }

    private void writeChunkDirectly(List<LoopData> chunk, WritableGroup chunksGroup, int chunkIndex, int matrixWidth) {
        float[][][] chunkArray = new float[chunk.size()][matrixWidth][matrixWidth];
        
        for (int i = 0; i < chunk.size(); i++) {
            chunkArray[i] = chunk.get(i).matrix;
        }
        
        String datasetName = "chunk_" + chunkIndex;
        chunksGroup.putDataset(datasetName, chunkArray);
        
        // Add to loop info lists
        synchronized (loopInfoLock) {
            for (LoopData loopData : chunk) {
                chr1List.add(loopData.chr1);
                start1List.add(loopData.start1);
                end1List.add(loopData.end1);
                chr2List.add(loopData.chr2);
                start2List.add(loopData.start2);
                end2List.add(loopData.end2);
                chunkIndexList.add(chunkIndex);
            }
        }
    }

    private void clearMatrix(float[][] matrix) {
        for (float[] row : matrix) {
            Arrays.fill(row, 0.0f);
        }
    }

    private float[][] copyMatrix(float[][] source) {
        int width = source.length;
        float[][] copy = new float[width][width];
        for (int i = 0; i < width; i++) {
            System.arraycopy(source[i], 0, copy[i], 0, width);
        }
        return copy;
    }

    private void printUsageAndExit() {
        System.out.println(usage);
        System.exit(19);
    }
}