/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Tiny graph runtime that can run graph
 *        containing only tvm PackedFunc.
 * \file graph_runtime.h
 */
#ifndef TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_

#include <string>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace runtime {

/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;
/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief operator attributes about tvm op */
struct TVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};


Module GraphRuntimeCreate(std::string sym_json,
                          tvm::runtime::Module m,
                          int device_type,
                          int device_id);


}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
