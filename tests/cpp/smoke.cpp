#include <continuum/ir/graph.hpp>

int main() {
  continuum::ir::Graph g;
  continuum::ir::Node n;
  g.add_node(n);
  return g.topo_order().empty() ? 1 : 0;
}
