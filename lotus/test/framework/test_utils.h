#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/ml_value.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif
#ifdef USE_TVM
#include "core/providers/nuphar/nuphar_execution_provider.h"
#endif  // USE_TVM

namespace onnxruntime {
namespace Test {
IExecutionProvider* TestCPUExecutionProvider();

#ifdef USE_CUDA
IExecutionProvider* TestCudaExecutionProvider();
#endif

#ifdef USE_TVM
IExecutionProvider* TestNupharExecutionProvider();
#endif  // USE_TVM

template <typename T>
void CreateMLValue(AllocatorPtr alloc,
                   const std::vector<int64_t>& dims,
                   const std::vector<T>& value,
                   MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  if (value.size() > 0) {
    memcpy(buffer, &value[0], element_type->Size() * shape.Size());
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
void AllocateMLValue(AllocatorPtr alloc,
                     const std::vector<int64_t>& dims,
                     MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}
}  // namespace Test
}  // namespace onnxruntime
