/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr.h
 * \brief Relay expression IR Node.
 */
#ifndef TVM_RELAY_EXPR_H_
#define TVM_RELAY_EXPR_H_

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/node.h>
#include <string>
#include "./base.h"
#include "./type.h"

namespace tvm {
namespace relay {
/*!
 * \brief Relay expression.
 */
class Expr;
/*!
 * \brief Base type of the Relay type hiearchy.
 */
class ExprNode : public RelayNode {
 public:
  /*!
   * \brief Stores the result of type inference(type checking).
   *
   * \note This can be undefined before type inference.
   *       this value is discarded during serialization.
   */
  Type checked_type_ = Type(nullptr);
  /*!
   * \return The checked_type
   */
  const Type& checked_type() const {
    CHECK(checked_type_.defined()) << "internal error: the type checker has "
                                      "not populated the checked_type "
                                   << "field for this node";
    return this->checked_type_;
  }

  static constexpr const char* _type_key = "relay.Expr";
  TVM_DECLARE_BASE_NODE_INFO(ExprNode, RelayNode);
};

RELAY_DEFINE_NODE_REF(Expr, ExprNode, NodeRef);

/*!
 * \brief Constant tensor, backed by an NDArray on cpu(0).
 *
 * \note scalar constants are represented by rank-0 const tensor.
 *  Constant folding are handled uniformly via Tensor types.
 */
class Constant;
/*!
 * \brief Constant tensor type.
 */
class ConstantNode : public ExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::NDArray data;

  // TODO(tqchen) add the function after we get TensorType constructor
  // TODO(tqchen) create simple TensorType constructor for concrete types.
  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return whether it is scalar(rank-0 tensor) */
  bool is_scalar() const { return data->ndim == 0; }

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("span", &span);
  }

  TVM_DLL static Constant make(runtime::NDArray data);

  static constexpr const char* _type_key = "relay.Constant";
  TVM_DECLARE_NODE_TYPE_INFO(ConstantNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Constant, ConstantNode, Expr);

/*! \brief Tuple of multiple Exprs */
class Tuple;
/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<relay::Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  TVM_DLL static Tuple make(tvm::Array<relay::Expr> fields);

  static constexpr const char* _type_key = "relay.Tuple";
  TVM_DECLARE_NODE_TYPE_INFO(TupleNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Tuple, TupleNode, Expr);

/*!
 * \brief Local variables used in the let expression.
 * This is similar to Var that is being used in the low level tensor expression.
 *
 * \note Each LocalVar is bind only once and is immutable/
 */
class LocalVar;
/*! \brief Container for LocalVar */
class LocalVarNode : public ExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  std::string name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name_hint", &name_hint);
  }

  TVM_DLL static LocalVar make(std::string name_hint);

  static constexpr const char* _type_key = "relay.LocalVar";
  TVM_DECLARE_NODE_TYPE_INFO(LocalVarNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(LocalVar, LocalVarNode, Expr);

/*!
 * \brief Global variable that leaves in the top-level environment.
 * This is used to enable recursive calls between function.
 *
 * \note GlobalVar can only corresponds to functions.
 */
class GlobalVar;
/*! \brief A GlobalId from the node's current type to target type. */
class GlobalVarNode : public ExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  std::string name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name_hint", &name_hint);
  }

  TVM_DLL static GlobalVar make(std::string name_hint);

  static constexpr const char* _type_key = "relay.GlobalVar";
  TVM_DECLARE_NODE_TYPE_INFO(GlobalVarNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(GlobalVar, GlobalVarNode, Expr);

/*!
 * \brief Function parameter declaration.
 */
class Param;
/*! \brief A parameter. */
class ParamNode : public ExprNode {
 public:
  /*! \brief The variable */
  LocalVar var;
  /*! \brief The type of the parameter */
  Type type;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("type", &type);
    v->Visit("span", &span);
  }

  TVM_DLL static Param make(LocalVar var, Type type);

  static constexpr const char* _type_key = "relay.Param";
  TVM_DECLARE_NODE_TYPE_INFO(ParamNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Param, ParamNode, Expr);

/*!
 * \brief Function (subgraph in computational graph)
 */
class Function;
/*! \brief Function container */
class FunctionNode : public ExprNode {
 public:
  /*! \brief Function parameters */
  tvm::Array<Param> params;
  /*! \brief User annotated return type of the function. */
  Type ret_type;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  Expr body;
  /*!
   * \brief Type parameters of the function.
   *  Enables the function to vary its type based on these.
   *  This corresponds to template paramaters in c++'s terminology.
   *
   * \note This can be usually empty for non-polymorphic functions.
   */
  tvm::Array<TypeParam> type_params;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("params", &params);
    v->Visit("ret_type", &ret_type);
    v->Visit("body", &body);
    v->Visit("type_params", &type_params);
    v->Visit("span", &span);
  }

  TVM_DLL static Function make(tvm::Array<Param> params, Type ret_type,
                               Expr body, tvm::Array<TypeParam> ty_params);

  static constexpr const char* _type_key = "relay.Function";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Function, FunctionNode, Expr);

// TODO(tqchen) change Expr to Attr after we introduce Attr system.
using Attrs = tvm::Map<std::string, Expr>;

/*!
 * \brief Call corresponds to operator invocation.
 *  Corresponds to the operator in computational graph terminology.
 */
class Call;
/*! \brief Call container. */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be relay::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, LocalVar).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The type arguments passed to polymorphic(template) function.
   *
   * This is the advance feature that is only used when the function is
   * polymorphic. It is safe to be ignored in most cases. For example, in the
   * following code, the type_args of addone call is [int].
   *
   * \code
   *
   * template<typename T>
   * T addone(T a) { return a + 1; }
   *
   * void main() {
   *   int x = addone<int>(10);
   * }
   *
   * \endcode
   */
  tvm::Array<Type> type_args;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("type_args", &type_args);
    v->Visit("span", &span);
  }

  TVM_DLL static Call make(Expr op, Array<Expr> args, Attrs attrs,
                           Array<Type> ty_args);

  static constexpr const char* _type_key = "relay.Call";
  TVM_DECLARE_NODE_TYPE_INFO(CallNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Call, CallNode, Expr);

/*!
 * \brief Let binding that binds a local var and optionally a type annotation.
 *
 * \note Let is useful to transform the program to be A-normal form.
 *  where each of the expression corresponds to a let binding.
 *
 *  For developers who are familar with the computational graph.
 *  Each of the let can be viewed as a operator node in the computational graph.
 *  Traversing the list of let bindings is similar to running
 * PostDFS-order(topo-order) traversal on the computational graph.
 */
class Let;
/*! \brief A binding of a sub-network. */
class LetNode : public ExprNode {
 public:
  /*! \brief The variable we bind to */
  LocalVar var;
  /*! \brief The value we bind var to */
  Expr value;
  /*! \brief The body of the let binding */
  Expr body;
  /*! \brief type annotation of value, this can be null */
  Type value_type;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("value_type", &value_type);
    v->Visit("span", &span);
  }

  TVM_DLL static Let make(LocalVar var, Expr value, Expr body, Type value_type);

  static constexpr const char* _type_key = "relay.Let";
  TVM_DECLARE_NODE_TYPE_INFO(LetNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Let, LetNode, Expr);

/*!
 * \brief Condition expression
 */
class If;
/*! \brief container of If */
class IfNode : public ExprNode {
 public:
  /*! \brief The condition */
  Expr cond;
  /*! \brief The value to take when condition is true */
  Expr true_value;
  /*! \brief The value to take when condition is false */
  Expr false_value;

  IfNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("cond", &cond);
    v->Visit("true_value", &true_value);
    v->Visit("false_value", &false_value);
    v->Visit("span", &span);
  }

  TVM_DLL static If make(Expr cond, Expr true_value, Expr false_value);

  static constexpr const char* _type_key = "relay.If";
  TVM_DECLARE_NODE_TYPE_INFO(IfNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(If, IfNode, Expr);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_H_
