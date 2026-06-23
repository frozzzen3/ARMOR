using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TVMEditor.Editing;
using TVMEditor.Editing.AffinityCalculation;
using TVMEditor.Editing.CenterDeformation;
using TVMEditor.Editing.SurfaceDeformation;
using TVMEditor.Editing.TransformPropagation;
using TVMEditor.IO;
using TVMEditor.Structures;

namespace TVMEditor.Test.Experiments
{
    public class center_affinity_deformation
    {
        public static void Run(string inputMeshDir, string inputCenterDir, string outputDir, int sourceIndex, int targetIndex)
        {
            Console.WriteLine(inputMeshDir);
            Console.WriteLine(inputCenterDir);
            Console.WriteLine(outputDir);
            var meshFiles = GetIndexedFiles(inputMeshDir, "*.obj");
            var centerFiles = GetIndexedFiles(inputCenterDir, "*.xyz");

            if (!meshFiles.TryGetValue(sourceIndex, out var sourceMeshPath))
            {
                throw new FileNotFoundException($"No source mesh found for index {sourceIndex:000} in {inputMeshDir}.");
            }

            if (!centerFiles.TryGetValue(sourceIndex, out var sourceCentersPath))
            {
                throw new FileNotFoundException($"No source centers found for index {sourceIndex:000} in {inputCenterDir}.");
            }

            Console.WriteLine($"Deforming sourceIndex={sourceIndex} to targetIndex={targetIndex}");
            var sequence = MeshIO.LoadSequenceFromObj(new[] { sourceMeshPath });
            var centers = CentersIO.LoadCentersFiles(new[] { sourceCentersPath });
            var (indices, transformations) = TransformationsIO.LoadIndexedTransformations(
                ResolveTransformationPath(inputCenterDir, $"indices_{sourceIndex:000}_{targetIndex:000}.txt"),
                ResolveTransformationPath(inputCenterDir, $"transformations_{sourceIndex:000}_{targetIndex:000}.txt"));

            var affinityCalculation = new DistanceDirectionAffinityCalculation
            {
                ShapeDistance = 1f
            };
            var centersDeformations = new AffinityCenterDeformation(affinityCalculation);
            var surfaceDeformation = new CustomSurfaceDeformation(affinityCalculation);
            var transformPropagation = new KabschTransformPropagation(affinityCalculation);

            var editor = new MeshEditor(affinityCalculation, centersDeformations, null, surfaceDeformation, transformPropagation);
            editor.Deform(sequence, centers, indices, transformations, 0, out var deformedSequence, out var deformedCenters);

            var outputPath = $"{outputDir}/output/deformed_{sourceIndex:000}_{targetIndex:000}.obj";
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            MeshIO.WriteMeshToObj(outputPath, deformedSequence.Meshes[0]);
            //MeshIO.WriteSequenceToObj($"{outputDir}/Dancer/dq", deformedSequence);
        }

        private static System.Collections.Generic.Dictionary<int, string> GetIndexedFiles(string directoryPath, string searchPattern)
        {
            return new DirectoryInfo(directoryPath)
                .GetFiles(searchPattern)
                .Where(file => file.Extension == ".obj" || file.Extension == ".bin" || file.Extension == ".xyz")
                .Select(file => new { file.FullName, Index = TryGetFileIndex(file.Name) })
                .Where(file => file.Index.HasValue)
                .ToDictionary(file => file.Index.Value, file => file.FullName);
        }

        private static int? TryGetFileIndex(string fileName)
        {
            var match = Regex.Match(fileName, @"(\d+)\.[^.]+$");
            return match.Success ? int.Parse(match.Groups[1].Value) : (int?)null;
        }

        private static string ResolveTransformationPath(string inputCenterDir, string fileName)
        {
            return Path.Combine(inputCenterDir, "transformation", fileName);
        }
    }
}
