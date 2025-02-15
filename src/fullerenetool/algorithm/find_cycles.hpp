#ifndef FIND_CYCLES_HPP
#define FIND_CYCLES_HPP

#include <vector>
#include <iostream>
#include <unordered_set>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/thread.hpp> // For hardware_concurrency

#include <parmcb/config.hpp>
#include <parmcb/wrappers.hpp>
#include <parmcb/parmcb.hpp>

#ifdef PARMCB_HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#endif

namespace cycle_finder
{

    class graph_cycle_finder
    {
    public:
        // Constructor
        graph_cycle_finder(int edge_num, const long *edges, int cores = 0);

        // Find and return cycles in the graph
        std::vector<std::vector<unsigned long>> find_and_print_cycles();

    private:
        typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, double>> Graph;
        typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;

        Graph g_;
        std::vector<std::vector<unsigned long>> cycles_;

        void compute_minimum_cycle_basis();
        void convert_cycles_to_python_format(const std::list<std::list<EdgeDescriptor>> &cpp_cycles);
        std::vector<unsigned long> build_node_path(const std::list<EdgeDescriptor> &cycle_edges);
        void set_global_tbb_concurrency(std::size_t cores = 0);
    };

    void graph_cycle_finder::set_global_tbb_concurrency(std::size_t cores)
    {
#ifdef PARMCB_HAVE_TBB
        // Set the number of cores to use (can be replaced with command line arguments if needed)
        if (cores == 0)
        {
            std::size_t cores = boost::thread::hardware_concurrency();
        }
        parmcb::set_global_tbb_concurrency(cores);
        std::cout << "[TBB]Using cores: " << cores << std::endl;
#endif
    }

    graph_cycle_finder::graph_cycle_finder(int edge_num, const long *edges, int cores)
    {
        // Initialize TBB concurrency settings
        if (cores == 0)
        {
            std::size_t cores = boost::thread::hardware_concurrency();
        }
        set_global_tbb_concurrency(cores);

        // Adding edges to the graph with default weights of 1.0
        for (int i = 0; i < edge_num; ++i)
        {
            boost::add_edge(edges[i * 2], edges[i * 2 + 1], 1.0, g_);
        }
    }

    void graph_cycle_finder::compute_minimum_cycle_basis()
    {
        typedef typename boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
        WeightMap weight_map = boost::get(boost::edge_weight, g_);

        std::list<std::list<EdgeDescriptor>> cycles_cpp;

        // double mcb_weight =
        parmcb::mcb_sva_iso_trees_tbb(g_, weight_map, std::back_inserter(cycles_cpp));

        // Convert C++ cycles to Python format
        convert_cycles_to_python_format(cycles_cpp);
    }

    void graph_cycle_finder::convert_cycles_to_python_format(const std::list<std::list<EdgeDescriptor>> &cpp_cycles)
    {

        for (const auto &cycle : cpp_cycles)
        {
            std::vector<unsigned long> cycle_py = build_node_path(cycle);
            cycles_.push_back(cycle_py);
        }
    }

    std::vector<unsigned long> graph_cycle_finder::build_node_path(const std::list<EdgeDescriptor> &cycle_edges)
    {
        std::vector<unsigned long> node_path;
        if (cycle_edges.empty())
        {
            return node_path;
        }

        // Start from the source of the first edge
        std::unordered_set<unsigned long> visited_nodes;
        auto it = cycle_edges.begin();

        unsigned long current_node = boost::source(*it, g_);
        node_path.push_back(current_node);
        visited_nodes.insert(current_node);

        current_node = boost::target(*it, g_);
        node_path.push_back(current_node);
        visited_nodes.insert(current_node);
        ++it;

        while (true)
        {
            bool found_next = false;
            for (auto edge_it = cycle_edges.begin(); edge_it != cycle_edges.end(); ++edge_it)
            {
                unsigned long u = boost::source(*edge_it, g_);
                unsigned long v = boost::target(*edge_it, g_);
                if (u == current_node && visited_nodes.find(v) == visited_nodes.end())
                {
                    current_node = v;
                    node_path.push_back(current_node);
                    visited_nodes.insert(current_node);
                    it = edge_it;
                    found_next = true;
                    break;
                }
                else if (v == current_node && visited_nodes.find(u) == visited_nodes.end())
                {
                    current_node = u;
                    node_path.push_back(current_node);
                    visited_nodes.insert(current_node);
                    it = edge_it;
                    found_next = true;
                    break;
                }
            }

            if (!found_next || current_node == node_path.front())
            {
                break;
            }
        }

        return node_path;
    }

    std::vector<std::vector<unsigned long>> graph_cycle_finder::find_and_print_cycles()
    {
        compute_minimum_cycle_basis();
        return cycles_;
    }

} // namespace cycle_finder

#endif // FIND_CYCLES_HPP
