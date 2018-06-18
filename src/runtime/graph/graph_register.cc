#include <tvm/runtime/registry.h>

#include "./graph_runtime.h"

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = GraphRuntimeCreate(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* mhandle = args[1];
    *rv = GraphRuntimeCreate(args[0],
                             *static_cast<tvm::runtime::Module*>(mhandle),
                             args[2], args[3]);
  });

} // runtime
} // tvm
