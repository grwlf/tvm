/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#ifndef TVM_RELAY_COMPILER_TYPE_FUNCTOR_H_
#define TVM_RELAY_COMPILER_TYPE_FUNCTOR_H_

#include <tvm/ir_functor.h>
#include "ir.h"

namespace tvm {
namespace relay {

template <typename FType>
class TypeFunctor;

// functions to be overriden.
#define TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }

#define RELAY_TYPE_FUNCTOR_DISPATCH(OP)                       \
  vtable.template set_dispatch<OP>(                           \
      [](const NodeRef& n, TSelf* self, Args... args) {       \
        return self->VisitType_(static_cast<const OP*>(n.node_.get()),    \
                                std::forward<Args>(args)...); \
      });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = tvm::IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) {
    return VisitType(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const TensorTypeNode* op,
                       Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeParamNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeConstraintNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
                       Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
                       Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeFunction* op, Args... args) TYPE_FUNCTOR_DEFAULT;
                       Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeCallNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;

  virtual R VisitTypeDefault_(const Node* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->type_key();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_TYPE_FUNCTOR_DISPATCH(TensorTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeParamNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeFunctionNode);
    RELAY_TYPE_FUNCTOR_DISPATCH(TypeCallNode);
    return vtable;
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_COMPILER_TYPE_FUNCTOR_H_