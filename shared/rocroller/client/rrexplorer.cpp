
#include <iostream>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visualizer.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

void rrexplorer(rocRoller::KernelGraph::KernelGraph const& original,
                std::string const&                         windowName,
                std::string const&                         asmFileName  = "",
                bool                                       tryColouring = true,
                bool                                       trySimplify  = true)
{
    auto graph = original;
    if(trySimplify)
    {
        removeRedundantBodyEdges(graph);
        removeRedundantSequenceEdges(graph);
    }

    rocRoller::KernelGraph::UnrollColouring       unrollColouring;
    rocRoller::KernelGraph::NaryArgumentColouring naryColouring;

    if(tryColouring)
    {
        // Try to get unroll colouring
        try
        {
            unrollColouring = rocRoller::KernelGraph::colourByUnrollValue(graph);
        }
        catch(...)
        {
            // Unroll colouring failed
        }

        // Try to get NaryArgument colouring
        try
        {
            naryColouring = rocRoller::KernelGraph::colourByNaryArgument(graph);
        }
        catch(...)
        {
            // NaryArgument colouring failed
        }
    }

    rocRoller::KernelGraph::visualize(
        graph, windowName, unrollColouring, naryColouring, asmFileName);
}

void rrexplorer(std::string yamlFileName, std::string asmFileName = "")
{
    std::string yaml;
    {
        std::stringstream ss;
        std::ifstream     file;
        file.open(yamlFileName);
        ss << file.rdbuf();
        file.close();
        yaml = ss.str();
    }

    auto graph = rocRoller::KernelGraph::fromYAML(yaml);
    rrexplorer(graph, yamlFileName, asmFileName);
}

void usage()
{
    std::cout << "Usage: rrexplorer <graph.yaml> [assembly.s]" << std::endl;
    std::cout << "  <graph.yaml>   - Required: Path to the kernel graph YAML file" << std::endl;
    std::cout << "  [assembly.s]   - Optional: Path to assembly file for visualization"
              << std::endl;
}

int main(int argc, const char* argv[])
{
    if(argc < 2 || argc > 3)
    {
        usage();
        return 1;
    }

    std::string yamlFileName = argv[1];
    std::string asmFileName  = (argc == 3) ? argv[2] : "";

    rrexplorer(yamlFileName, asmFileName);
}
