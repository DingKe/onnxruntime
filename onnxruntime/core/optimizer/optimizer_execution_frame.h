// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/execution_frame.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

class OptimizerExecutionFrame final : public IExecutionFrame {
 public:
  class Info {
   public:
    Info(const InitializedTensorSet& initialized_tensor_set, const std::vector<const Node*>& nodes);

    AllocatorPtr GetAllocator(const OrtAllocatorInfo& info) const {
      if (info.id != device_id_ || info.mem_type != mem_type_) {
        return nullptr;
      }
      return allocator_ptr_;
    }

    AllocatorPtr GetAllocator() const {
      return allocator_ptr_;
    }

    const MLValueNameIdxMap& GetMLValueNameIdxMap() const noexcept { return mlvalue_name_idx_map_; }
    const std::unordered_map<int, const NodeArg*>& GetMLValueIdxNodeArgMap() const noexcept { return mlvalue_idx_nodearg_map_; }
    const std::unordered_map<int, MLValue>& GetInitializers() const noexcept { return initializers_; }
    const NodeIndexInfo& GetNodeIndexInfo() const { return *node_index_info_; }

   private:
    // The optimizer is running on CPU execution provider by default.
    std::unique_ptr<CPUExecutionProvider> cpu_execution_provider_;
    const int device_id_{0};
    const OrtMemType mem_type_{OrtMemTypeDefault};
    AllocatorPtr allocator_ptr_;

    // MLValues for optimizer
    MLValueNameIdxMap mlvalue_name_idx_map_;
    std::unordered_map<int, const NodeArg*> mlvalue_idx_nodearg_map_;
    std::unordered_map<int, MLValue> initializers_;
    std::unique_ptr<NodeIndexInfo> node_index_info_;
  };

  OptimizerExecutionFrame(const Info& param,
                          const std::vector<int>& feed_mlvalue_idxs,
                          const std::vector<MLValue>& feeds,
                          const std::vector<int>& fetch_mlvalue_idxs,
                          std::vector<MLValue>& fetches);

  ~OptimizerExecutionFrame();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerExecutionFrame);

  AllocatorPtr GetAllocatorImpl(const OrtAllocatorInfo& info) const override;
  Status CreateNodeOutputMLValueImpl(MLValue& mlvalue, int mlvalue_idx, const TensorShape* shape) override;

  const Info& info_;
};

}  // namespace onnxruntime