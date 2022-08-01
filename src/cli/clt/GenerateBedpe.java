package cli.clt;

import cli.Main;
import cli.utils.GenericLocusParser;
import cli.utils.flags.Anchor;
import cli.utils.flags.LoopGenerator;
import javastraw.feature1D.GenomeWide1DList;
import javastraw.feature2D.Feature2D;
import javastraw.feature2D.Feature2DList;
import javastraw.reader.basics.Chromosome;
import javastraw.reader.basics.ChromosomeHandler;
import javastraw.reader.basics.ChromosomeTools;
import javastraw.tools.ParallelizationTools;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class GenerateBedpe {

    public static void run(String[] args, CommandLineParser parser) {
        // generate <forward.bed> <reverse.bed> <min> <max> <genome> <outname>

        if (args.length != 7) {
            Main.printGeneralUsageAndExit(4);
        }
        String forwardMotifFile = args[1];
        String reverseMotifFile = args[2];
        long minDist = Long.parseLong(args[3]);
        long maxDist = Long.parseLong(args[4]);
        String genomeID = args[5];
        String outname = args[6];

        ChromosomeHandler handler = ChromosomeTools.loadChromosomes(genomeID);

        int percentile = parser.getPercentileOption(-1);

        GenomeWide1DList<Anchor> forwardAnchors = GenericLocusParser.loadFromBEDFile(handler, forwardMotifFile, percentile, true);
        GenomeWide1DList<Anchor> reverseAnchors = GenericLocusParser.loadFromBEDFile(handler, reverseMotifFile, percentile, true);
        System.out.println("Number of anchors: " + forwardAnchors.size() + " - " + reverseAnchors.size());

        Feature2DList output = createLoops(handler, forwardAnchors, reverseAnchors, minDist, maxDist);
        System.out.println(outname + ".bedpe");
        output.exportFeatureList(new File(outname + ".bedpe"), false, Feature2DList.ListFormat.NA);
        System.out.println("generation complete");
    }

    private static Feature2DList createLoops(ChromosomeHandler handler, GenomeWide1DList<Anchor> forwardAnchors,
                                             GenomeWide1DList<Anchor> reverseAnchors,
                                             long minDist, long maxDist) {
        Feature2DList output = new Feature2DList();
        Chromosome[] chromosomes = handler.getChromosomeArrayWithoutAllByAll();

        AtomicInteger index = new AtomicInteger(0);
        ParallelizationTools.launchParallelizedCode(() -> {
            int currIndex = index.getAndIncrement();
            while (currIndex < chromosomes.length) {
                Chromosome chromosome = chromosomes[currIndex];
                if (forwardAnchors.size() > 0 && reverseAnchors.size() > 0) {
                    List<Feature2D> newLoops = generate(chromosome, forwardAnchors, reverseAnchors, minDist, maxDist);
                    if (newLoops.size() > 0) {
                        synchronized (output) {
                            output.addByKey(Feature2DList.getKey(chromosome, chromosome), newLoops);
                        }
                    }
                }
                currIndex = index.getAndIncrement();
            }
        });
        return output;
    }

    private static List<Feature2D> generate(Chromosome chromosome, GenomeWide1DList<Anchor> forwardAnchors,
                                            GenomeWide1DList<Anchor> reverseAnchors, long minGenomeDist, long maxGenomeDist) {

        List<Anchor> forwards = forwardAnchors.getFeatures("" + chromosome.getIndex());
        forwards.sort(Comparator.comparingInt(Anchor::getMid));

        List<Anchor> reverses = reverseAnchors.getFeatures("" + chromosome.getIndex());
        reverses.sort(Comparator.comparingInt(Anchor::getMid));

        List<Feature2D> results = new ArrayList<>();

        for (Anchor forward : forwards) {
            for (Anchor reverse : reverses) {
                if (forward.getEnd() < reverse.getStart()) {
                    Feature2D feature2D = LoopGenerator.createIntraFeature(chromosome, forward, reverse,
                            minGenomeDist, maxGenomeDist);
                    if (feature2D != null) {
                        results.add(feature2D);
                    }
                }
            }
        }
        return results;
    }
}