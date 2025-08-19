#include <SDL.h>
#include <SDL_opengl.h>

#include <backends/imgui_impl_opengl2.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>
#include <imgui_node_editor.h>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <fstream>
#include <regex>
#include <sstream>

/*******************************************************************************
 * ImGUI helpers
 */

namespace ImNE = ax::NodeEditor;

namespace ImGui
{
    void TextUnformatted(std::string const t)
    {
        TextUnformatted(t.c_str());
    }

    ImVec2 CalcTextSize(std::string const t)
    {
        return CalcTextSize(t.c_str());
    }

    bool Combo(std::string const& label, int* selected, std::vector<std::string>& items)
    {
        return ImGui::Combo(
            label.c_str(),
            selected,
            [](void* vec, int idx, const char** out_text) -> bool {
                auto v = reinterpret_cast<std::vector<std::string>*>(vec);
                if(idx < 0 || idx >= v->size())
                    return false;
                *out_text = v->at(idx).c_str();
                return true;
            },
            reinterpret_cast<void*>(&items),
            items.size());
    }
}

/*******************************************************************************
 * rocRoller KernelGraph visualizer
 */

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Visualizer
        {
            using namespace ControlGraph;
            using namespace CoordinateGraph;

            /*******************************************************************************
             * Type definitions and forward declarations
             */

            struct Flags
            {
                bool visible   = true;
                bool collapsed = false;
            };

            using FlagMap = std::map<int, Flags>; //< Tag -> Flags
            using GraphColouring
                = std::map<int, std::map<int, int>>; //< Tag -> Coordinate -> Colour/Value
            using GraphLayout = std::map<int, std::pair<float, float>>; //< Tag -> (x, y)

            /*******************************************************************************
             * Colour schemes
             */

            std::vector<ImColor> ColourBrewer2Divergent8 = {
                {140, 81, 10},
                {191, 129, 45},
                {223, 194, 125},
                {246, 232, 195},
                {199, 234, 229},
                {128, 205, 193},
                {53, 151, 143},
                {1, 102, 94},
            };

            std::vector<ImColor> ColourBrewer2Qualitative12 = {
                {166, 206, 227},
                {31, 120, 180},
                {178, 223, 138},
                {51, 160, 44},
                {251, 154, 153},
                {227, 26, 28},
                {253, 191, 111},
                {255, 127, 0},
                {202, 178, 214},
                {106, 61, 154},
                {255, 255, 153},
                {177, 89, 40},
            };

            /*******************************************************************************
             * Layout helpers
             */

            namespace Layout
            {
                GraphLayout graphvizGraphLayout(auto const&    graph,
                                                FlagMap const& flags  = {},
                                                int            xScale = 2,
                                                int            yScale = -1)
                {
                    Log::trace("graphvizGraphLayout: Starting layout computation");
                    GraphLayout rv;

                    constexpr int maxNameLength = 64;
                    char          name[maxNameLength];
                    std::snprintf(name, maxNameLength - 1, "tmp");

                    Log::trace("graphvizGraphLayout: Opening graphviz graph");
                    auto g = agopen(name, Agdirected, nullptr);

                    std::map<int, Agnode_t*> nodes;
                    std::map<int, Agedge_t*> edges;

                    // Add rr-nodes and rr-edges as gv-nodes
                    Log::trace("graphvizGraphLayout: Adding nodes to graphviz graph");
                    for(auto elem : graph.allElements())
                    {
                        bool add = flags.empty() || flags.at(elem).visible;
                        if(add)
                        {
                            std::snprintf(name, maxNameLength - 1, "%d", elem);
                            nodes[elem] = agnode(g, name, 1);
                        }
                    }
                    Log::trace("graphvizGraphLayout: Added {} nodes", nodes.size());
                    // Add gv-edges between rr-nodes and rr-edges
                    Log::trace("graphvizGraphLayout: Adding edges to graphviz graph");
                    for(auto elem : graph.allElements())
                    {
                        auto downstream = graph.getNeighbours(elem, Graph::Direction::Downstream);
                        for(auto down : downstream)
                        {
                            bool add = flags.empty()
                                       || (flags.at(elem).visible && flags.at(down).visible);
                            if(add)
                            {
                                std::snprintf(name, maxNameLength - 1, "%d", elem * 1000000 + down);
                                agedge(g, nodes[elem], nodes[down], name, 1);
                            }
                        }
                    }
                    Log::trace("graphvizGraphLayout: Finished adding edges");

                    Log::trace("graphvizGraphLayout: Creating graphviz context");
                    auto c = gvContext();
                    Log::trace("graphvizGraphLayout: Computing layout (this may take a while)...");
                    gvLayout(c, g, "dot");
                    Log::trace("graphvizGraphLayout: Layout computed, rendering...");
                    gvRender(c, g, "dot", nullptr);
                    Log::trace("graphvizGraphLayout: Render done, extracting coordinates");
                    for(auto elem : graph.allElements())
                    {
                        if(nodes.contains(elem))
                        {
                            auto coord = ND_coord(nodes[elem]);
                            rv[elem]   = {xScale * coord.x, yScale * coord.y};
                        }
                    }
                    Log::trace("graphvizGraphLayout: Cleaning up graphviz resources");
                    gvFreeLayout(c, g);
                    gvFinalize(c);
                    agclose(g);

                    Log::trace(
                        "graphvizGraphLayout: Layout computation complete, returning {} positions",
                        rv.size());
                    return rv;
                }

                /**
                 * Compute graph layout.
                 */
                GraphLayout coordinateGraphLayout(
                    rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph const& graph,
                    FlagMap const&                                                  flags)
                {
                    return graphvizGraphLayout(graph, flags, 2, -2);
                }

                GraphLayout controlGraphLayout(
                    rocRoller::KernelGraph::ControlGraph::ControlGraph const& graph,
                    FlagMap const&                                            flags)
                {
                    return graphvizGraphLayout(graph, flags, 2, -1);
                }
            }

            /*******************************************************************************
             * Python code generation helpers
             */

            namespace Python
            {
                std::string makePythonFunction(KernelGraph const&               graph,
                                               std::string const&               functionName,
                                               std::vector<int> const&          required,
                                               Expression::ExpressionPtr const& expr)
                {
                    std::string docstring;

                    auto arguments = std::vector<Expression::ExpressionPtr>{};
                    for(int i = 0; i < required.size(); ++i)
                    {
                        arguments.push_back(std::make_shared<rocRoller::Expression::Expression>(
                            Expression::PositionalArgument(
                                i, Register::Type::Scalar, DataType::UInt32)));
                        docstring.append(
                            fmt::format("    # {} = {} tag {}\n",
                                        rocRoller::Expression::toPython(arguments.back()),
                                        toString(graph.coordinates.getNode(required[i])),
                                        required[i]));
                    }

                    std::string argumentExtract = "    def PositionalArgument(i):\n"
                                                  "        return args[i]\n";

                    std::string functionBody
                        = fmt::format("def {}(*args):\n{}\n{}\n    return {}\n",
                                      functionName,
                                      docstring,
                                      argumentExtract,
                                      rocRoller::Expression::toPython(expr));

                    return functionBody;
                }
            }

            /*******************************************************************************
             * UI helper functions
             */

            int colourByUnrollCombo(int& selectedColour, GraphColouring const& colouring)
            {
                std::vector<int>         availableUnrollCoords       = {-1};
                std::vector<std::string> availableUnrollCoordsLabels = {"None"};
                {
                    std::set<int> availableUnrollCoordsSet;
                    for(auto [tag, colours] : colouring)
                    {
                        for(auto [coord, value] : colours)
                            availableUnrollCoordsSet.insert(coord);
                    }
                    for(auto x : availableUnrollCoordsSet)
                    {
                        availableUnrollCoords.push_back(x);
                        availableUnrollCoordsLabels.push_back(std::to_string(x));
                    }
                }

                ImGui::TextUnformatted("Colour by: ");
                ImGui::SameLine();
                ImGui::Combo(" unroll coordinate", &selectedColour, availableUnrollCoordsLabels);

                return availableUnrollCoords[selectedColour];
            }

            template <typename T>
            std::string join(std::string const& separator, T const& xs)
            {
                if(xs.empty())
                    return "";

                std::vector<int> vxs;
                for(auto x : xs)
                    vxs.push_back(x);

                std::string rv;
                for(auto i = 0; i < vxs.size() - 1; ++i)
                    rv = rv + std::to_string(vxs[i]) + separator;
                rv = rv + std::to_string(vxs.back());
                return rv;
            }

            /*******************************************************************************
             * GraphVizualizer class definition
             */

            struct GraphVizualizer
            {
                enum Attribute
                {
                    START, //< Coordinate is tagged as a 'start' coordinate (for path finding).
                    END, //< Coordinate is tagged as an 'end' coordinate (for path finding).
                    PATH, //< Coordinate is in the path between the 'start' and 'end' coordinates.
                    STORAGE, //< Coordinate is a "storage" node
                    HARDWARE, //< Coordinate is a "hardware" node
                    LOOPISH, //< Coordinate is a "loopish" node

                    BODY, //< Control edge is a `Body` edge
                    SEQUENCE, //< Control edge is a `Sequence` edge

                    CONNECTED, //< Coordinate is connected to a selected Operation

                    CLICKED, //< Node was clicked
                };

                using AttributeMap = std::map<int, std::unordered_set<Attribute>>;
                using ColourMap
                    = std::map<Attribute, std::vector<std::pair<ImNE::StyleColor, ImVec4>>>;

                ColourMap colours;

                // Global
                bool m_showControlGraph    = true;
                bool m_showCoordinateGraph = false;

                // Coordinate graph
                bool m_showDataFlowEdges       = false;
                bool m_showIndexingEdges       = false;
                bool m_showTransformEdges      = true;
                bool m_showDanglingCoordinates = false;
                bool m_showLoadPath            = true;
                bool m_showStorePath           = true;
                bool m_showRequiredPath        = false;
                int  m_showCoordinateTag       = -1;

                std::vector<int> m_selectedCoordinates, m_startCoordinates, m_endCoordinates,
                    m_targetCoordinates;

                GraphLayout m_coordinateLayout;

                // Control graph
                int m_showOperationTag = -1;

                std::set<int> m_selectedOperations;

                bool m_elideControlEdges = true;

                int m_highlightOperation  = -1;
                int m_scrolledToOperation = -1;

                int  m_selectedUnrollColour = 0;
                bool m_showNaryArgColouring = false;

                FlagMap m_controlFlags;

                GraphLayout                  m_operationLayout;
                std::unordered_map<int, int> m_bodyParents;
                std::set<int>                m_inLoadPath, m_inStorePath;

                std::string m_pythonFunction;

                // Assembly viewer
                bool                            m_showAssemblyWindow = false;
                std::vector<std::string>        m_asmLines;
                std::map<int, std::vector<int>> m_tagToAsmLines;
                int                             m_asmFontSizeIndex = 2; // Default to 14pt
                std::set<int>                   m_previousSelectedOperations;

                /**
                 * Load and parse assembly file.
                 */
                void loadAssemblyFile(std::string const& asmFileName)
                {
                    if(asmFileName.empty())
                        return;

                    std::ifstream file(asmFileName);
                    if(!file.is_open())
                    {
                        Log::warn("Could not open assembly file: {}", asmFileName);
                        return;
                    }

                    m_asmLines.clear();
                    m_tagToAsmLines.clear();

                    // Regex to match "; (op TAG)" at end of line
                    std::regex tagPattern(R"(;\s*\(op\s+(\d+)\)\s*$)");

                    std::string line;
                    int         lineNum = 0;
                    while(std::getline(file, line))
                    {
                        m_asmLines.push_back(line);

                        // Check if line ends with "; (op TAG)"
                        std::smatch match;
                        if(std::regex_search(line, match, tagPattern))
                        {
                            int tag = std::stoi(match[1].str());
                            m_tagToAsmLines[tag].push_back(lineNum);
                        }
                        lineNum++;
                    }

                    m_showAssemblyWindow = !m_asmLines.empty();
                    Log::info("Loaded {} lines from assembly file", m_asmLines.size());
                }

                /**
                 * Pre-compute default layouts.
                 */
                void computeLayouts(KernelGraph const& graph)
                {
                    m_coordinateLayout = Layout::coordinateGraphLayout(graph.coordinates, {});
                    m_operationLayout  = Layout::controlGraphLayout(graph.control, {});

                    // XXX
                    //m_bodyParents = ControlFlowRWTracer(graph).getBodyParents();

                    std::set<int> storageNodes;
                    for(auto elem : graph.coordinates.allElements())
                    {
                        if(isStorageCoordinate(elem, graph))
                            storageNodes.insert(elem);
                    }

                    // Precompute m_inLoadPath and m_inStorePath.
                    for(auto elem : storageNodes)
                    {
                        auto x = findRequiredCoordinates(elem, Graph::Direction::Downstream, graph)
                                     .second;
                        x.erase(elem);
                        std::copy(
                            x.cbegin(), x.cend(), std::inserter(m_inLoadPath, m_inLoadPath.end()));

                        auto y = findRequiredCoordinates(elem, Graph::Direction::Upstream, graph)
                                     .second;
                        y.erase(elem);
                        std::copy(y.cbegin(),
                                  y.cend(),
                                  std::inserter(m_inStorePath, m_inStorePath.end()));
                    }
                }

                template <typename N, typename T>
                std::string
                    toDescription(N const& node, int tag, T const& graph, KernelGraph const& kgraph)
                {
                    bool constexpr isControl
                        = std::is_same_v<T, rocRoller::KernelGraph::ControlGraph::ControlGraph>;

                    if(isControl)
                    {
                        // XXX Make this a visitor?

                        auto maybeAssign = kgraph.control.get<Assign>(tag);
                        if(maybeAssign)
                        {
                            auto dest = kgraph.mapper.get(tag, NaryArgument::DEST);
                            auto elem = kgraph.coordinates.getElement(dest);
                            return fmt::format(
                                "Assign: DF({}) = {}\nDF({}): {}",
                                dest,
                                toString(maybeAssign->expression),
                                dest,
                                std::visit([&](auto const& x) { return toString(x); }, elem));
                        }

                        auto maybeSetCoordinate = kgraph.control.get<SetCoordinate>(tag);
                        if(maybeSetCoordinate)
                        {
                            auto dest = kgraph.mapper.getConnections(tag)[0].coordinate;
                            return fmt::format(
                                "SetCoordinate: DF({}) = {}\nDF({}): {}",
                                dest,
                                toString(maybeSetCoordinate->value),
                                dest,
                                toString(kgraph.coordinates.getNode<Dimension>(dest)));
                        }

                        auto maybeDeallocate = kgraph.control.get<Deallocate>(tag);
                        if(maybeDeallocate)
                        {
                            std::vector<int> tags;
                            for(auto const& c : kgraph.mapper.getConnections(tag))
                            {
                                tags.push_back(c.coordinate);
                            }
                            return fmt::format(
                                "Deallocate: Coords {}; Args {}", tags, maybeDeallocate->arguments);
                        }
                    }

                    return toString(node);
                }

                /**
                 * Render a graph (generic).
                 */
                template <typename T>
                std::map<int, int> renderGraph(T const&              graph,
                                               KernelGraph const&    kgraph,
                                               FlagMap const&        flags,
                                               AttributeMap const&   attributes,
                                               GraphColouring const& colouring,
                                               int                   showColourOf = -1)
                {
                    std::map<int, int> widths;

                    bool constexpr isControl
                        = std::is_same_v<T, rocRoller::KernelGraph::ControlGraph::ControlGraph>;

                    auto getOutputAttrId = [](int elem) { return (1 << 30) + elem; };
                    auto getInputAttrId  = [](int elem) { return (1 << 29) + elem; };

                    ImGui::PushFont(m_font);

                    ImNE::PushStyleVar(ImNE::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
                    ImNE::PushStyleVar(ImNE::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));

                    for(auto elem : graph.allElements())
                    {
                        if(!flags.contains(elem))
                        {
                            std::cout << "WARNING: missing flags " << elem << std::endl;
                            continue;
                        }

                        if(!flags.at(elem).visible)
                            continue;

                        auto type = graph.getElementType(elem);
                        if(type == Graph::ElementType::Node)
                        {
                            auto node = graph.getNode(elem);

                            int styleColourCount = 0;
                            if(attributes.contains(elem))
                            {
                                for(auto attr : attributes.at(elem))
                                {
                                    for(auto [colourVar, colour] : colours.at(attr))
                                    {
                                        ImNE::PushStyleColor(colourVar, colour);
                                        ++styleColourCount;
                                    }
                                }
                            }
                            if(showColourOf != -1 && colouring.contains(elem))
                            {
                                if(colouring.at(elem).contains(showColourOf))
                                {
                                    auto colourVar   = ImNE::StyleColor::StyleColor_NodeBg;
                                    auto colourValue = colouring.at(elem).at(showColourOf);

                                    // Cycle through colours and darken if we exceed available colours
                                    int numColours  = ColourBrewer2Qualitative12.size();
                                    int colourIndex = colourValue % numColours;
                                    int cycle       = colourValue / numColours;

                                    ImColor baseColour = ColourBrewer2Qualitative12.at(colourIndex);
                                    ImVec4  colour     = ImVec4(baseColour.Value.x,
                                                           baseColour.Value.y,
                                                           baseColour.Value.z,
                                                           baseColour.Value.w);

                                    // Darken the color based on cycle count
                                    if(cycle > 0)
                                    {
                                        float darkenFactor = 1.0f / (1.0f + cycle * 0.3f);
                                        colour.x *= darkenFactor;
                                        colour.y *= darkenFactor;
                                        colour.z *= darkenFactor;
                                    }

                                    ImNE::PushStyleColor(colourVar, colour);
                                    ++styleColourCount;
                                }
                            }

                            ImNE::BeginNode(elem);

                            std::string nodeTextHeader;
                            if constexpr(isControl)
                            {
                                nodeTextHeader = std::format("{}: {}", elem, name(node));
                            }
                            else
                            {
                                auto getStrideNoAssert = [](const auto x) {
                                    return std::visit([](const auto a) { return a.stride; }, x);
                                };
                                nodeTextHeader = std::format("{}: {}\nS: {}\nJ: {}",
                                                             elem,
                                                             name(node),
                                                             toString(getSize(node)),
                                                             toString(getStrideNoAssert(node)));
                            }

                            auto nodeTextHeaderSize = ImGui::CalcTextSize(nodeTextHeader);

                            widths[elem] = nodeTextHeaderSize.x;

                            ImNE::BeginPin(getInputAttrId(elem), ImNE::PinKind::Input);
                            ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));
                            ImNE::EndPin();

                            ImGui::TextUnformatted(nodeTextHeader);

                            ImNE::BeginPin(getOutputAttrId(elem), ImNE::PinKind::Output);
                            ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));
                            ImNE::EndPin();

                            ImNE::EndNode();

                            if(ImGui::IsItemHovered())
                            {
                                auto description = toDescription(node, elem, graph, kgraph);
                                ImGui::SetTooltip("%s", description.c_str());
                            }

                            ImNE::PopStyleColor(styleColourCount);
                        }
                        else if(type == Graph::ElementType::Edge)
                        {
                            auto edge = graph.getEdge(elem);
                            if constexpr(isControl)
                            {
                                auto isBody     = graph.template get<Body>(elem).has_value();
                                auto isSequence = graph.template get<Sequence>(elem).has_value();

                                if((isBody || isSequence) && m_elideControlEdges)
                                    continue;
                            }

                            ImNE::PushStyleVar(ImNE::StyleVar_NodeRounding, 0.f);

                            std::string nodeTextHeader
                                = std::format("{}: {}", elem, toString(edge));
                            if constexpr(!isControl)
                            {
                                auto inOrder
                                    = graph.getNeighbours(elem, Graph::Direction::Upstream);
                                auto outOrder
                                    = graph.getNeighbours(elem, Graph::Direction::Downstream);
                                if(!inOrder.empty())
                                {
                                    nodeTextHeader += "\nInputs: ";
                                    for(auto i = 0; i < inOrder.size(); ++i)
                                    {
                                        nodeTextHeader
                                            += std::format("{}{}",
                                                           inOrder[i],
                                                           (i == inOrder.size() - 1) ? "" : ", ");
                                    }
                                }
                                if(!inOrder.empty())
                                {
                                    nodeTextHeader += "\nOutputs: ";
                                    for(auto i = 0; i < outOrder.size(); ++i)
                                    {
                                        nodeTextHeader
                                            += std::format("{}{}",
                                                           outOrder[i],
                                                           (i == outOrder.size() - 1) ? "" : ", ");
                                    }
                                }
                            }

                            auto nodeTextHeaderSize = ImGui::CalcTextSize(nodeTextHeader);

                            widths[elem] = nodeTextHeaderSize.x;

                            ImNE::BeginNode(elem);

                            ImNE::BeginPin(getInputAttrId(elem), ImNE::PinKind::Input);
                            ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));

                            ImNE::EndPin();

                            ImGui::TextUnformatted(nodeTextHeader);

                            ImNE::BeginPin(getOutputAttrId(elem), ImNE::PinKind::Output);
                            ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));

                            ImNE::EndPin();

                            ImNE::EndNode();

                            ImNE::PopStyleVar(1);
                        }
                    }

                    ImNE::PopStyleVar(2);
                    ImGui::PopFont();

                    int linkId = 0;
                    for(auto elem : graph.getEdges())
                    {
                        if(!flags.contains(elem))
                        {
                            std::cout << "WARNING: missing flags " << elem << std::endl;
                            continue;
                        }

                        if(!flags.at(elem).visible)
                            continue;

                        bool isBody     = false;
                        bool isSequence = false;
                        if constexpr(std::is_same_v<
                                         T,
                                         rocRoller::KernelGraph::ControlGraph::ControlGraph>)
                        {
                            isBody     = graph.template get<Body>(elem).has_value();
                            isSequence = graph.template get<Sequence>(elem).has_value();
                        }

                        bool elide  = (isSequence || isBody) && m_elideControlEdges;
                        auto colour = ImColor(255, 255, 255);

                        if(isBody)
                            colour = colours.at(Attribute::BODY)[0].second;
                        if(isSequence)
                            colour = colours.at(Attribute::SEQUENCE)[0].second;

                        if(!elide)
                        {
                            for(auto input : graph.getNeighbours(elem, Graph::Direction::Upstream))
                            {

                                ImNE::Link(
                                    linkId++, getOutputAttrId(input), getInputAttrId(elem), colour);
                            }
                            for(auto output :
                                graph.getNeighbours(elem, Graph::Direction::Downstream))
                            {
                                ImNE::Link(linkId++,
                                           getOutputAttrId(elem),
                                           getInputAttrId(output),
                                           colour);
                            }
                        }
                        else
                        {
                            auto parent
                                = *only(graph.getNeighbours(elem, Graph::Direction::Upstream));
                            auto child
                                = *only(graph.getNeighbours(elem, Graph::Direction::Downstream));
                            ImNE::Link(
                                linkId++, getOutputAttrId(parent), getInputAttrId(child), colour);
                        }
                    }

                    return widths;
                }

                /**
                 * Render the coordinate graph.
                 */
                void renderCoordinateGraph(KernelGraph const&                 graph,
                                           GraphColouring const&              unrollColouring,
                                           std::map<int, NaryArgument> const& naryColouring)
                {
                    ImGui::Begin("CoordinateTransform graph");

                    //
                    // Controls
                    //

                    auto reflow = ImGui::Button("Reflow");
                    ImGui::SameLine();
                    ImGui::Checkbox("DataFlow", &m_showDataFlowEdges);
                    ImGui::SameLine();
                    ImGui::Checkbox("Indexing", &m_showIndexingEdges);
                    ImGui::SameLine();
                    ImGui::Checkbox("CoordinateTransform", &m_showTransformEdges);
                    ImGui::SameLine();
                    ImGui::Checkbox("Show danglers", &m_showDanglingCoordinates);
                    ImGui::SameLine();
                    ImGui::Checkbox("Show load path", &m_showLoadPath);
                    ImGui::SameLine();
                    ImGui::Checkbox("Show store path", &m_showStorePath);
                    ImGui::SameLine();
                    ImGui::Checkbox("Show required path only", &m_showRequiredPath);

                    const int   inputLength = 32;
                    static char showTagInput[inputLength];
                    if(ImGui::Button("Move to tag: "))
                        m_showCoordinateTag = std::stoi(showTagInput);
                    else
                        m_showCoordinateTag = -1;
                    ImGui::SameLine();
                    ImGui::InputText("", showTagInput, inputLength);

                    ImGui::TextUnformatted("Selected tags: ");
                    ImGui::SameLine();
                    ImGui::TextUnformatted(join(", ", m_selectedCoordinates));

                    if(ImGui::Button("Start tags: "))
                    {
                        m_startCoordinates = m_selectedCoordinates;
                        m_targetCoordinates.clear();
                    }
                    ImGui::SameLine();
                    ImGui::TextUnformatted(join(", ", m_startCoordinates));

                    if(ImGui::Button("End tags:   "))
                    {
                        m_endCoordinates = m_selectedCoordinates;
                        m_targetCoordinates.clear();
                    }
                    ImGui::SameLine();
                    ImGui::TextUnformatted(join(", ", m_endCoordinates));

                    if(ImGui::Button("Required:   "))
                    {
                        m_targetCoordinates = m_selectedCoordinates;
                        m_startCoordinates.clear();
                        m_endCoordinates.clear();
                    }
                    ImGui::SameLine();
                    ImGui::TextUnformatted(join(", ", m_targetCoordinates));

                    auto makePythonLoad  = ImGui::Button("Make Python load function");
                    auto makePythonStore = ImGui::Button("Make Python store function");

                    //
                    // Compute path
                    //

                    std::unordered_set<int> path;
                    if(!m_startCoordinates.empty() && !m_endCoordinates.empty())
                    {
                        path = graph.coordinates
                                   .path<Graph::Direction::Downstream>(m_startCoordinates,
                                                                       m_endCoordinates)
                                   .to<std::unordered_set>();

                        if(path.empty())
                        {
                            path = graph.coordinates
                                       .path<Graph::Direction::Upstream>(m_startCoordinates,
                                                                         m_endCoordinates)
                                       .to<std::unordered_set>();
                        }
                    }

                    if(!m_targetCoordinates.empty())
                    {
                        // XXX loop
                        auto store = findRequiredCoordinates(
                            m_targetCoordinates.back(), Graph::Direction::Upstream, graph);
                        auto load = findRequiredCoordinates(
                            m_targetCoordinates.back(), Graph::Direction::Downstream, graph);

                        path = store.second;
                        std::copy(load.second.cbegin(),
                                  load.second.cend(),
                                  std::inserter(path, path.end()));
                    }

                    if(makePythonLoad)
                    {
                        auto target = m_targetCoordinates.back();
                        auto [required, _path]
                            = findRequiredCoordinates(target, Graph::Direction::Downstream, graph);
                        auto arguments = std::vector<Expression::ExpressionPtr>{};
                        for(int i = 0; i < required.size(); ++i)
                        {
                            auto node = graph.coordinates.getNode(required[i]);
                            arguments.push_back(std::make_shared<rocRoller::Expression::Expression>(
                                Expression::PositionalArgument(
                                    i, Register::Type::Scalar, DataType::UInt32)));
                        }
                        auto expr        = graph.coordinates.reverse(arguments, {target}, required);
                        m_pythonFunction = Python::makePythonFunction(
                            graph, fmt::format("load{}", target), required, expr[0]);

                        ImGui::SetClipboardText(m_pythonFunction.c_str());
                        ImGui::OpenPopup("Python function");
                    }

                    if(makePythonStore)
                    {
                        auto target = m_targetCoordinates.back();
                        auto [required, _path]
                            = findRequiredCoordinates(target, Graph::Direction::Upstream, graph);
                        auto arguments = std::vector<Expression::ExpressionPtr>{};
                        for(int i = 0; i < required.size(); ++i)
                        {
                            arguments.push_back(std::make_shared<rocRoller::Expression::Expression>(
                                Expression::PositionalArgument(
                                    i, Register::Type::Scalar, DataType::UInt32)));
                        }
                        auto expr        = graph.coordinates.forward(arguments, required, {target});
                        m_pythonFunction = Python::makePythonFunction(
                            graph, fmt::format("store{}", target), required, expr[0]);

                        ImGui::SetClipboardText(m_pythonFunction.c_str());
                        ImGui::OpenPopup("Python function");
                    }

                    if(ImGui::BeginPopupModal(
                           "Python function", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                    {
                        ImGui::TextUnformatted("Python function (copied to clipboard):");
                        ImGui::Separator();
                        ImGui::TextUnformatted(m_pythonFunction);
                        if(ImGui::Button("Close"))
                            ImGui::CloseCurrentPopup();
                        ImGui::EndPopup();
                    }

                    //
                    // Compute visible
                    //
                    AttributeMap attributes;

                    FlagMap flags;
                    for(auto elem : graph.coordinates.allElements())
                    {
                        auto type = graph.coordinates.getElementType(elem);
                        if(type == Graph::ElementType::Node)
                        {
                            flags[elem].visible = true;

                            if(!m_showLoadPath && m_inLoadPath.contains(elem)
                               && !m_inStorePath.contains(elem))
                                flags[elem].visible = false;

                            if(!m_showStorePath && !m_inLoadPath.contains(elem)
                               && m_inStorePath.contains(elem))
                                flags[elem].visible = false;

                            if(m_showRequiredPath && !path.contains(elem))
                                flags[elem].visible = false;

                            if(isStorageCoordinate(elem, graph))
                                attributes[elem].insert(Attribute::STORAGE);
                            if(isHardwareCoordinate(elem, graph))
                                attributes[elem].insert(Attribute::HARDWARE);
                            if(isLoopishCoordinate(elem, graph))
                                attributes[elem].insert(Attribute::LOOPISH);
                        }
                    }

                    // Make sure everything attached via DataFlow to a PATH node is visible.
                    if(m_showRequiredPath)
                    {
                        for(auto elem : path)
                        {
                            auto type = graph.coordinates.getElementType(elem);
                            if(type != Graph::ElementType::Node)
                                continue;

                            for(auto x :
                                graph.coordinates.getNeighbours(elem, Graph::Direction::Upstream))
                            {
                                if(isEdge<DataFlowEdge>(graph.coordinates.getEdge(x)))
                                {
                                    for(auto tag : graph.coordinates.getNeighbours(
                                            x, Graph::Direction::Upstream))

                                        flags[tag].visible = true;
                                }
                            }
                        }
                    }

                    for(auto elem : graph.coordinates.getEdges())
                    {
                        auto type = graph.coordinates.getElementType(elem);
                        if(type == Graph::ElementType::Edge)
                        {
                            flags[elem].visible = false;

                            auto edge = graph.coordinates.getEdge(elem);
                            flags[elem].visible |= m_showDataFlowEdges && isEdge<DataFlow>(edge);
                            flags[elem].visible |= m_showIndexingEdges && isEdge<DataFlowEdge>(edge)
                                                   && !isEdge<DataFlow>(edge);
                            flags[elem].visible
                                |= m_showTransformEdges && isEdge<CoordinateTransformEdge>(edge);

                            if(flags[elem].visible)
                            {
                                bool missingUpstream = true;
                                for(auto x : graph.coordinates.getNeighbours(
                                        elem, Graph::Direction::Upstream))
                                {
                                    if(flags[x].visible)
                                        missingUpstream = false;
                                }
                                bool missingDownstream = true;
                                for(auto x : graph.coordinates.getNeighbours(
                                        elem, Graph::Direction::Downstream))
                                {
                                    if(flags[x].visible)
                                        missingDownstream = false;
                                }
                                if(missingDownstream || missingUpstream)
                                    flags[elem].visible = false;
                            }
                        }
                    }

                    if(!m_showDanglingCoordinates)
                    {
                        for(auto elem : graph.coordinates.allElements())
                        {
                            auto type = graph.coordinates.getElementType(elem);
                            if(type == Graph::ElementType::Node)
                            {
                                if(!flags[elem].visible)
                                    continue;

                                bool visible = false;
                                for(auto x : graph.coordinates.getNeighbours(
                                        elem, Graph::Direction::Upstream))
                                {
                                    if(flags[x].visible)
                                        visible = true;
                                }
                                for(auto x : graph.coordinates.getNeighbours(
                                        elem, Graph::Direction::Downstream))
                                {
                                    if(flags[x].visible)
                                        visible = true;
                                }
                                flags[elem].visible = visible;
                            }
                        }
                    }

                    for(auto elem : path)
                        attributes[elem].insert(Attribute::PATH);

                    // For each selected operation, add the Attribute::CONNECTED attribute
                    for(auto opTag : m_selectedOperations)
                    {
                        for(auto conn : graph.mapper.getConnections(opTag))
                        {
                            attributes[conn.coordinate].insert(Attribute::CONNECTED);
                        }
                    }

                    // Merge both colourings into a combined GraphColouring
                    GraphColouring combinedColouring = unrollColouring;
                    int            showColourOf      = -1;

                    if(m_showNaryArgColouring)
                    {
                        // For NaryArgument colouring, use a fixed key
                        // (0) and store the NaryArgument value
                        //
                        // This allows us to colour all nodes by their
                        // respective NaryArgument values
                        for(auto const& [tag, naryArg] : naryColouring)
                        {
                            combinedColouring[tag][0] = static_cast<int>(naryArg);
                        }
                        showColourOf = 0;
                    }

                    ImNE::SetCurrentEditor(m_coordinateContext);
                    ImNE::Begin("Coordinate Graph");
                    auto widths = renderGraph(graph.coordinates,
                                              graph,
                                              flags,
                                              attributes,
                                              combinedColouring,
                                              showColourOf);

                    if(reflow)
                    {
                        m_coordinateLayout
                            = Layout::coordinateGraphLayout(graph.coordinates, flags);
                        for(auto [elem, xy] : m_coordinateLayout)
                        {
                            if(flags[elem].visible)
                                ImNE::SetNodePosition(elem,
                                                      {xy.first - 0.5f * widths[elem], xy.second});
                        }
                    }

                    if(m_showCoordinateTag > 0)
                    {
                        ImNE::SelectNode(m_showCoordinateTag);
                        ImNE::NavigateToSelection();
                    }

                    int selectedObjectCount = ImNE::GetSelectedObjectCount();
                    if(selectedObjectCount > 0)
                    {
                        std::vector<ImNE::NodeId> selectedNodes(selectedObjectCount);

                        auto numSelectedNodes
                            = ImNE::GetSelectedNodes(selectedNodes.data(), selectedObjectCount);
                        m_selectedCoordinates.resize(numSelectedNodes);
                        for(auto i = 0; i < numSelectedNodes; ++i)
                        {
                            m_selectedCoordinates[i] = selectedNodes[i].Get();
                        }
                    }
                    else
                    {
                        m_selectedCoordinates.clear();
                    }

                    ImNE::End();
                    ImNE::SetCurrentEditor(nullptr);

                    ImGui::End();
                }

                /**
                 * Render control graph.
                 */
                void renderControlGraph(KernelGraph const&                 graph,
                                        GraphColouring const&              unrollColouring,
                                        std::map<int, NaryArgument> const& naryColouring)
                {
                    ImGui::Begin("ControlFlow graph");

                    //
                    // Controls
                    //

                    auto reflow = ImGui::Button("Reflow");

                    ImGui::SameLine();
                    if(ImGui::Button("Uncollapse all"))
                    {
                        for(auto& [tag, flags] : m_controlFlags)
                            flags.collapsed = false;
                    }

                    ImGui::SameLine();
                    ImGui::Checkbox("Elide edges", &m_elideControlEdges);

                    ImGui::SameLine();
                    ImGui::TextUnformatted("Selected tags: ");
                    ImGui::SameLine();
                    ImGui::TextUnformatted(join(", ", m_selectedOperations));

                    auto showUnrollColourOf
                        = colourByUnrollCombo(m_selectedUnrollColour, unrollColouring);

                    ImGui::SameLine();
                    ImGui::Checkbox("Colour by NaryArgument", &m_showNaryArgColouring);

                    // Merge both colourings into a combined GraphColouring
                    GraphColouring combinedColouring = unrollColouring;
                    int            showColourOf      = showUnrollColourOf;

                    if(m_showNaryArgColouring)
                    {
                        // For NaryArgument colouring, use a fixed key
                        // (0) and store the NaryArgument value
                        //
                        // This allows us to colour all nodes by their
                        // respective NaryArgument values
                        for(auto const& [tag, naryArg] : naryColouring)
                        {
                            combinedColouring[tag][0] = static_cast<int>(naryArg);
                        }
                        showColourOf = 0;
                    }

                    // move this to speedbar
                    const int   inputLength = 32;
                    static char showTagInput[inputLength];
                    if(ImGui::Button("Move to tag: "))
                        m_showOperationTag = std::stoi(showTagInput);
                    ImGui::SameLine();
                    ImGui::InputText("", showTagInput, inputLength);

                    //
                    // Visibility
                    //

                    // First pass: default visible
                    for(auto elem : graph.control.allElements())
                    {
                        m_controlFlags[elem].visible = true;
                    }

                    // Second pass: collapse bodies (nodes).
                    for(auto elem : graph.control.allElements())
                    {
                        auto type = graph.control.getElementType(elem);
                        if(type != Graph::ElementType::Node)
                            continue;

                        auto target = elem;
                        while(true)
                        {
                            if(m_controlFlags[m_bodyParents[target]].collapsed)
                            {
                                m_controlFlags[elem].visible = false;
                                break;
                            }
                            auto parent = m_bodyParents[target];
                            if(m_bodyParents.contains(parent))
                            {
                                target = parent;
                                continue;
                            }
                            break;
                        }
                    }

                    // Third pass: hide dangling edges (edges).
                    for(auto elem : graph.control.allElements())
                    {
                        auto type = graph.control.getElementType(elem);
                        if(type != Graph::ElementType::Edge)
                            continue;

                        bool show = false;
                        for(auto x : graph.control.getNeighbours(elem, Graph::Direction::Upstream))
                        {
                            if(m_controlFlags[x].visible)
                                show = true;
                        }
                        bool hasDownstream = false;
                        for(auto x :
                            graph.control.getNeighbours(elem, Graph::Direction::Downstream))

                        {
                            if(m_controlFlags[x].visible)
                            {
                                hasDownstream = true;
                                show          = true;
                            }
                        }

                        m_controlFlags[elem].visible = show && hasDownstream;
                    }

                    //
                    // Draw
                    //

                    AttributeMap attributes;
                    ImNE::SetCurrentEditor(m_controlContext);
                    ImNE::Begin("Control Flow Graph");

                    renderGraph(graph.control,
                                graph,
                                m_controlFlags,
                                attributes,
                                combinedColouring,
                                showColourOf);

                    if(reflow)
                    {
                        m_operationLayout
                            = Layout::controlGraphLayout(graph.control, m_controlFlags);
                        for(auto [elem, xy] : m_operationLayout)
                        {
                            if(m_controlFlags[elem].visible)
                                ImNE::SetNodePosition(elem, {xy.first, xy.second});
                        }
                    }

                    if(m_showOperationTag > 0)
                    {
                        ImNE::SelectNode(m_showOperationTag);
                        ImNE::NavigateToSelection();
                    }
                    m_showOperationTag = -1;

                    int selectedObjectCount = ImNE::GetSelectedObjectCount();
                    if(selectedObjectCount > 0)
                    {
                        std::vector<ImNE::NodeId> selectedNodes(selectedObjectCount);

                        auto numSelectedNodes
                            = ImNE::GetSelectedNodes(selectedNodes.data(), selectedObjectCount);
                        m_selectedOperations.clear();
                        for(auto i = 0; i < numSelectedNodes; ++i)
                        {
                            m_selectedOperations.insert(selectedNodes[i].Get());
                        }
                    }
                    else
                    {
                        m_selectedOperations.clear();
                    }

                    int doubleClickedOperation = ImNE::GetDoubleClickedNode().Get();
                    if(doubleClickedOperation > 0)
                    {
                        m_highlightOperation = doubleClickedOperation;
                    }

                    ImNE::End();
                    ImNE::SetCurrentEditor(nullptr);

                    ImGui::End();

                    //
                    // Speedbar
                    //

                    ImGui::Begin("Operations");
                    auto operationTags = graph.control.getNodes().to<std::set>();
                    for(auto elem : operationTags)
                    {
                        auto node = graph.control.getNode(elem);

                        std::string label;
                        {
                            std::stringstream ss;
                            ss << std::setw(6) << elem << ": " << name(node);
                            label = ss.str();
                        }

                        ImGui::PushID(elem);
                        ImGui::Checkbox("", &m_controlFlags[elem].collapsed);

                        bool highlight = m_selectedOperations.contains(elem);
                        if(highlight)
                            ImGui::PushStyleColor(ImGuiCol_Text, colours[CLICKED][0].second);

                        ImGui::SameLine();
                        ImGui::TextUnformatted(label);

                        if(ImGui::IsItemHovered())
                        {
                            if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_G)))
                                m_showOperationTag = elem;

                            if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_U)))
                                if(m_bodyParents.contains(elem))
                                {
                                    m_showOperationTag   = m_bodyParents[elem];
                                    m_highlightOperation = m_bodyParents[elem];
                                }

                            if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_C)))
                                m_controlFlags[elem].collapsed = !m_controlFlags[elem].collapsed;
                        }

                        if(highlight)
                            ImGui::PopStyleColor();

                        if(elem == m_highlightOperation && elem != m_scrolledToOperation)
                        {
                            m_scrolledToOperation = elem;
                            ImGui::SetScrollHereY();
                        }

                        ImGui::PopID();
                    }
                    ImGui::End();
                }

                /**
                 * Render assembly viewer window.
                 */
                void renderAssemblyWindow()
                {
                    ImGui::Begin("Assembly Viewer", &m_showAssemblyWindow);

                    // Font size selector
                    std::vector<std::string> fontSizeLabels;
                    for(int size : m_monoFontSizes)
                    {
                        fontSizeLabels.push_back(std::to_string(size) + "pt");
                    }
                    ImGui::TextUnformatted("Font size:");
                    ImGui::SameLine();
                    ImGui::Combo("##AsmFontSize", &m_asmFontSizeIndex, fontSizeLabels);

                    ImGui::Separator();

                    // Get lines to highlight for all selected operations
                    std::set<int> highlightSet;
                    for(int selectedOp : m_selectedOperations)
                    {
                        if(m_tagToAsmLines.contains(selectedOp))
                        {
                            const auto& lines = m_tagToAsmLines.at(selectedOp);
                            highlightSet.insert(lines.begin(), lines.end());
                        }
                    }

                    // Detect selection change and find first matching line
                    bool selectionChanged  = (m_selectedOperations != m_previousSelectedOperations);
                    int  firstMatchingLine = -1;
                    if(selectionChanged && !highlightSet.empty())
                    {
                        firstMatchingLine            = *highlightSet.begin();
                        m_previousSelectedOperations = m_selectedOperations;
                    }

                    // Use selected monospace font
                    if(m_asmFontSizeIndex >= 0 && m_asmFontSizeIndex < m_monoFonts.size()
                       && m_monoFonts[m_asmFontSizeIndex] != nullptr)
                    {
                        ImGui::PushFont(m_monoFonts[m_asmFontSizeIndex]);
                    }

                    // Use child window for scrolling
                    ImGui::BeginChild("AsmScrollRegion",
                                      ImVec2(0, 0),
                                      false,
                                      ImGuiWindowFlags_HorizontalScrollbar);

                    // Calculate line height for scrolling
                    float lineHeight = ImGui::GetTextLineHeightWithSpacing();

                    // Display each line
                    for(int i = 0; i < m_asmLines.size(); ++i)
                    {
                        // Scroll to first matching line when selection changes
                        if(i == firstMatchingLine)
                        {
                            // Scroll so the line is roughly in the middle of the view
                            float targetY = i * lineHeight - ImGui::GetWindowHeight() * 0.3f;
                            ImGui::SetScrollY(std::max(0.0f, targetY));
                        }

                        bool isHighlighted = highlightSet.contains(i);

                        if(isHighlighted)
                        {
                            // Draw dim dark blue background behind the text
                            ImVec2 textSize  = ImGui::CalcTextSize(m_asmLines[i].c_str());
                            ImVec2 cursorPos = ImGui::GetCursorScreenPos();
                            ImGui::GetWindowDrawList()->AddRectFilled(
                                cursorPos,
                                ImVec2(cursorPos.x + ImGui::GetContentRegionAvail().x,
                                       cursorPos.y + lineHeight),
                                IM_COL32(40, 60, 100, 255)); // Dim dark blue background
                        }

                        ImGui::TextUnformatted(m_asmLines[i]);
                    }

                    ImGui::EndChild();

                    if(m_asmFontSizeIndex >= 0 && m_asmFontSizeIndex < m_monoFonts.size()
                       && m_monoFonts[m_asmFontSizeIndex] != nullptr)
                    {
                        ImGui::PopFont();
                    }

                    ImGui::End();
                }

                /**
                 * Render the full GUI.
                 *
                 * Backend agnostic.
                 */
                void renderGUI(KernelGraph const&           graph,
                               UnrollColouring const&       unrollColouring,
                               NaryArgumentColouring const& naryColouring)
                {
                    ImGui::Begin("Controls");
                    ImGui::Checkbox("Show ControlFlow graph", &m_showControlGraph);
                    ImGui::Checkbox("Show CoordinateTransform graph", &m_showCoordinateGraph);
                    if(!m_asmLines.empty())
                        ImGui::Checkbox("Show Assembly Viewer", &m_showAssemblyWindow);
                    ImGui::End();

                    if(m_showCoordinateGraph)
                        renderCoordinateGraph(graph,
                                              unrollColouring.coordinateColour,
                                              naryColouring.coordinateColour);

                    if(m_showControlGraph)
                        renderControlGraph(
                            graph, unrollColouring.operationColour, naryColouring.operationColour);

                    if(m_showAssemblyWindow && (not m_asmLines.empty()))
                        renderAssemblyWindow();
                }

                /**
                 * Setup colours.
                 *
                 * Backend agnostic.
                 */
                void setupTheme()
                {
                    colours[GraphVizualizer::Attribute::STORAGE]
                        = {{ImNE::StyleColor_NodeBg, ImColor(200, 200, 20)}};

                    colours[GraphVizualizer::Attribute::HARDWARE]
                        = {{ImNE::StyleColor_NodeBg, ImColor(20, 200, 20)}};

                    colours[GraphVizualizer::Attribute::LOOPISH]
                        = {{ImNE::StyleColor_NodeBg, ImColor(20, 20, 200)}};

                    colours[GraphVizualizer::Attribute::PATH]
                        = {{ImNE::StyleColor_NodeBg, ImColor(255, 0, 0)}};

                    colours[GraphVizualizer::Attribute::CONNECTED]
                        = {{ImNE::StyleColor_NodeBg, ImColor(230, 176, 50)}};

                    colours[GraphVizualizer::Attribute::BODY]
                        = {{ImNE::StyleColor_Count, ImColor(255, 0, 0)}};

                    colours[GraphVizualizer::Attribute::SEQUENCE]
                        = {{ImNE::StyleColor_Count, ImColor(128, 255, 128)}};

                    colours[GraphVizualizer::Attribute::CLICKED]
                        = {{ImNE::StyleColor_Count, ImColor(255, 176, 50, 255)}};
                }

                /**
                 * Setup the GUI.
                 *
                 * Specific to SDL2 + OpenGL2.
                 */
                void setupGUI(std::string const& windowName)
                {
                    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
                        return;

                    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
                    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
                    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
                    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
                    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
                    SDL_WindowFlags window_flags
                        = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
                                            | SDL_WINDOW_ALLOW_HIGHDPI);

                    m_window = SDL_CreateWindow(windowName.c_str(),
                                                SDL_WINDOWPOS_CENTERED,
                                                SDL_WINDOWPOS_CENTERED,
                                                1280,
                                                720,
                                                window_flags);
                    if(m_window == nullptr)
                    {
                        Log::warn("Error: SDL_CreateWindow(): {}", SDL_GetError());
                        return;
                    }

                    m_gl_context = SDL_GL_CreateContext(m_window);
                    SDL_GL_MakeCurrent(m_window, m_gl_context);
                    SDL_GL_SetSwapInterval(1);

                    IMGUI_CHECKVERSION();
                    ImGui::CreateContext();
                    ImGuiIO& io    = ImGui::GetIO();
                    io.IniFilename = "rr_graph.ini";
                    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
                    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

                    ImGui_ImplSDL2_InitForOpenGL(m_window, m_gl_context);
                    ImGui_ImplOpenGL2_Init();

                    // Generic
                    m_coordinateConfig.SelectButtonIndex   = 0;
                    m_coordinateConfig.DragButtonIndex     = 0;
                    m_coordinateConfig.NavigateButtonIndex = 1;
                    m_coordinateConfig.EnableSmoothZoom    = true;
                    m_coordinateConfig.SettingsFile        = "rr_graph_coordinates.json";

                    m_controlConfig.SelectButtonIndex   = 0;
                    m_controlConfig.DragButtonIndex     = 0;
                    m_controlConfig.NavigateButtonIndex = 1;
                    m_controlConfig.EnableSmoothZoom    = true;
                    m_controlConfig.SettingsFile        = "rr_graph_control.json";

                    m_coordinateContext = ImNE::CreateEditor(&m_coordinateConfig);
                    m_controlContext    = ImNE::CreateEditor(&m_controlConfig);

                    io.Fonts->AddFontDefault();
                    //                m_font = io.Fonts->AddFontFromFileTTF("inconsolata.ttf", 96);
                    m_font = io.Fonts->AddFontFromFileTTF(
                        "/usr/share/texlive/texmf-dist/fonts/truetype/public/dejavu/"
                        "DejaVuSans.ttf",
                        18);

                    // Load monospace fonts for assembly viewer at different sizes
                    const char* monoFontPath
                        = "/usr/share/texlive/texmf-dist/fonts/truetype/public/"
                          "dejavu/DejaVuSansMono.ttf";
                    for(int size : m_monoFontSizes)
                    {
                        ImFont* font
                            = io.Fonts->AddFontFromFileTTF(monoFontPath, static_cast<float>(size));
                        m_monoFonts.push_back(font);
                    }
                }

                /**
                 * GUI main loop.
                 *
                 * Specific to SDL2 + OpenGL2.
                 */
                void renderLoop(KernelGraph const&           graph,
                                UnrollColouring const&       unrollColouring,
                                NaryArgumentColouring const& naryColouring)
                {
                    ImGuiIO& io = ImGui::GetIO();

                    bool done = false;
                    while(!done)
                    {
                        SDL_Event event;
                        while(SDL_PollEvent(&event))
                        {
                            ImGui_ImplSDL2_ProcessEvent(&event);
                            if(event.type == SDL_QUIT)
                                done = true;
                            if(event.type == SDL_WINDOWEVENT
                               && event.window.event == SDL_WINDOWEVENT_CLOSE
                               && event.window.windowID == SDL_GetWindowID(m_window))
                                done = true;
                        }

                        ImGui_ImplOpenGL2_NewFrame();
                        ImGui_ImplSDL2_NewFrame();
                        ImGui::NewFrame();

                        renderGUI(graph, unrollColouring, naryColouring);

                        ImGui::Render();
                        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
                        glClear(GL_COLOR_BUFFER_BIT);
                        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
                        SDL_GL_SwapWindow(m_window);
                    }
                }

                /**
                 * Shutdown GUI.
                 *
                 * Specific to SDL2 + OpenGL2.
                 */
                void shutdownGUI()
                {
                    // Generic
                    ImNE::DestroyEditor(m_coordinateContext);
                    ImNE::DestroyEditor(m_controlContext);

                    // Cleanup; backend specific
                    ImGui_ImplOpenGL2_Shutdown();
                    ImGui_ImplSDL2_Shutdown();
                    ImGui::DestroyContext();

                    SDL_GL_DeleteContext(m_gl_context);
                    SDL_DestroyWindow(m_window);
                    SDL_Quit();
                }

                ImNE::Config         m_coordinateConfig;
                ImNE::EditorContext* m_coordinateContext;

                ImNE::Config         m_controlConfig;
                ImNE::EditorContext* m_controlContext;

                SDL_Window*   m_window;
                SDL_GLContext m_gl_context;

                ImFont*              m_font;
                std::vector<ImFont*> m_monoFonts;
                std::vector<int>     m_monoFontSizes = {10, 12, 14, 16, 18, 20, 24};
            };

            /*******************************************************************************
             * Public interface functions
             */

            // Might want to allow this to accept arguments so that
            // transforms can highlight nodes etc
#ifdef ROCROLLER_ENABLE_IMGUI
            void visualizeImpl(KernelGraph const&           graph,
                               std::string const&           windowName,
                               UnrollColouring const&       unrollColouring,
                               NaryArgumentColouring const& naryColouring,
                               std::string const&           asmFileName)
            {
                AssertFatal(sizeof(ImDrawIdx) >= 4,
                            "ImGUI configuration mismatch: ImDrawIdx should be at least 32bit for "
                            "rocRoller graphs.");
                GraphVizualizer viz;
                viz.setupGUI(windowName);
                viz.setupTheme();
                viz.loadAssemblyFile(asmFileName);
                viz.computeLayouts(graph);
                viz.renderLoop(graph, unrollColouring, naryColouring);
                viz.shutdownGUI();
            }
#else
            void visualizeImpl(KernelGraph const&           graph,
                               std::string const&           windowName,
                               UnrollColouring const&       unrollColouring,
                               NaryArgumentColouring const& naryColouring,
                               std::string const&           asmFileName)
            {
                Log::warn("rocRoller::KernelGraph::visualize called but ImGUI disabled.");
            }
#endif

        } // namespace Visualizer

        /*******************************************************************************
         * Public KernelGraph::visualize() wrapper function
         */

        void visualize(KernelGraph const&           graph,
                       std::string const&           windowName,
                       UnrollColouring const&       unrollColouring,
                       NaryArgumentColouring const& naryColouring,
                       std::string const&           asmFileName)
        {
            Visualizer::visualizeImpl(
                graph, windowName, unrollColouring, naryColouring, asmFileName);
        }

    }
}
