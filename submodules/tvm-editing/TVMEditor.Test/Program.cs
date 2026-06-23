using System;
using System.IO;
using TVMEditor.Test.Experiments;
using System.Diagnostics;

namespace TVMEditor.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 5)
            {
                Console.WriteLine("Usage: Program <sourceIndex> <targetIndex> <inputMeshDir> <inputCenterDir> <outputDir>");
                Console.WriteLine("<sourceIndex>: Source frame index to deform");
                Console.WriteLine("<targetIndex>: Target frame index to deform toward");
                Console.WriteLine("<inputMeshDir>: Input directory containing meshes");
                Console.WriteLine("<inputCenterDir>: Input directory containing centers and transformation folder");
                Console.WriteLine("<outputDir>: Output directory for deformed meshes");
                return;
            }
            if (!int.TryParse(args[0], out int sourceIndex))
            {
                Console.WriteLine("Error: <sourceIndex> must be an integer.");
                return;
            }

            if (!int.TryParse(args[1], out int targetIndex))
            {
                Console.WriteLine("Error: <targetIndex> must be an integer.");
                return;
            }

            string inputMeshDir = args[2];
            string inputCenterDir = args[3];
            string outputDir = args[4];

            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            Stopwatch stopwatch = Stopwatch.StartNew();

            center_affinity_deformation.Run(inputMeshDir, inputCenterDir, outputDir, sourceIndex, targetIndex);

            stopwatch.Stop();
            Console.WriteLine($"Elapsed time: {stopwatch.Elapsed.TotalSeconds:F2} seconds");
        }
    }
}
