#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(LotusIR::Graph* graph, const std::string& provider)
      : graph_(graph), copy_node_count_(0), provider_(provider) {}

  bool ModifyGraph();

 private:
  void ProcessDefs(const LotusIR::Node& node);
  void AddCopyNode(const ConstPointerContainer<std::vector<LotusIR::NodeArg*>>& args, bool is_input);

 private:
  std::set<const LotusIR::Node*> provider_nodes_;
  std::set<const LotusIR::NodeArg*> non_provider_defs_;     // all input/output defs of non-provider nodes
  std::set<const LotusIR::NodeArg*> provider_input_defs_;   // all input defs of provider nodes
  std::set<const LotusIR::NodeArg*> provider_output_defs_;  // all output defs of provider nodes
  std::map<LotusIR::NodeArg*, LotusIR::NodeArg*> replacements_;
  int copy_node_count_;
  LotusIR::Graph* graph_;
  std::string provider_;
};

}  // namespace Lotus