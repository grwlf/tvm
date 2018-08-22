"""
Test the type unifier, which solves systems of equations
between incomplete types.
"""
import tvm
from tvm.relay import ir
from tvm.relay.unifier import UnionFind, TypeUnifier
from tvm.relay.ir_builder import bool_type, uint_type, int_type, float_type, func_type
from tvm.relay import ir_builder as build
import tvm.relay.make as mk


def test_insert_and_find():
    uf = mk.UnionFind()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    uf.insert(v1)
    uf.insert(v2)
    assert uf.find(v1) == v1
    assert uf.find(v2) == v2


def test_insert_error():
    uf = mk.UnionFind()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    uf.insert(v1)
    try:
        uf.find(v2)
        assert False
    except:
        return


def test_unify():
    uf = mk.UnionFind()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    v3 = mk.IncompleteType(ir.Kind.Type)
    uf.insert(v1)
    uf.insert(v2)
    uf.insert(v3)
    uf.unify(v1, v2)
    rep = uf.find(v1)
    assert (rep == v1 or rep == v2)
    assert uf.find(v1) == rep
    assert uf.find(v2) == rep
    assert uf.find(v3) == v3
    assert v3 != rep
    uf.unify(v1, v3)
    new_rep = uf.find(v3)
    assert (rep == v1 or rep == v2 or rep == v3)
    assert uf.find(v1) == new_rep
    assert uf.find(v2) == new_rep
    assert uf.find(v3) == new_rep


def test_unify_multiple_levels():
    uf = mk.UnionFind()
    v = [mk.IncompleteType(ir.Kind.Type) for _ in range(9)]
    for var in v:
        uf.insert(var)
    uf.unify(v[0], v[1])
    uf.unify(v[0], v[2])
    uf.unify(v[3], v[4])
    uf.unify(v[4], v[5])
    uf.unify(v[6], v[7])
    uf.unify(v[6], v[8])
    rep1 = uf.find(v[0])
    rep2 = uf.find(v[3])
    rep3 = uf.find(v[6])
    assert (rep1 == v[0] or rep1 == v[1] or rep1 == v[2])
    assert (rep2 == v[3] or rep2 == v[4] or rep2 == v[5])
    assert (rep3 == v[6] or rep3 == v[7] or rep3 == v[8])
    for i in range(3):
        assert uf.find(v[i]) == rep1
        assert uf.find(v[i + 3]) == rep2
        assert uf.find(v[i + 6]) == rep3
    # now unify two of the groups
    uf.unify(v[1], v[4])
    new_rep1 = uf.find(v[0])
    new_rep2 = uf.find(v[6])
    assert (new_rep1 == v[0] or new_rep1 == v[1] or new_rep1 == v[2]
            or new_rep1 == v[3] or new_rep1 == v[4] or new_rep1 == v[5])
    assert (new_rep2 == v[6] or new_rep2 == v[7] or new_rep2 == v[8])
    for i in range(6):
        assert uf.find(v[i]) == new_rep1
    for i in range(3):
        assert uf.find(v[i + 6]) == new_rep2

# We have checked that the basic machinery in the UnionFind works
# and now we will test the type unifier which will fill in holes
# between type equalities by the process of unification.


def unify_types(t1, t2):
    unifier = mk.TypeUnifier()
    return unifier.unify(t1, t2)

# TODO(sslyu, weberlo, joshpoll): put in isinstance asserts once those work


def test_unify_int():
    intty = int_type(1)
    unified = unify_types(intty, intty)
    assert intty == unified


def test_unify_bool():
    boolty = bool_type()
    unified = unify_types(boolty, boolty)
    assert boolty == unified


def test_unify_float():
    floatty = float_type(4)
    unified = unify_types(floatty, floatty)
    assert floatty == unified


def test_unify_incompatible_basetypes():
    bt = bool_type()
    intty = int_type(32)
    try:
        unify_types(bt, intty)
        assert False
    except:
        return


def test_unify_concrete_func_type():
    arr1 = func_type([int_type()], int_type())
    arr2 = func_type([int_type()], int_type())
    unified = unify_types(arr1, arr2)
    assert unified == arr1


def test_unify_func_type_with_holes():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.BaseType)
    unifier.insert(v1)
    unifier.unify(v1, bool_type())
    arr1 = func_type([int_type()], bool_type())
    arr2 = func_type([int_type()], v1)
    unified = unifier.unify(arr1, arr2)
    assert unified == arr1

    v2 = mk.IncompleteType(ir.Kind.BaseType)
    unifier.insert(v2)
    unifier.unify(v2, int_type())
    arr3 = func_type([v2], bool_type())
    unified = unifier.unify(arr1, arr3)
    assert unified == arr1


def test_reject_incompatible_func_types():
    arr1 = func_type([int_type()], bool_type())
    arr2 = func_type([int_type(), bool_type()], bool_type())
    try:
        unify_types(arr1, arr2)
        assert False
    except:
        return

# def test_unify_concrete_type_quantifiers():
#     tq1 = TypeQuantifier(TypeParam("id1", ir.Kind.Type), int_type())
#     tq2 = TypeQuantifier(TypeParam("id2", ir.Kind.Type), int_type())
#     unified = unify_types(tq1, tq2)
#     assert unified == tq1

# def test_unify_basetype_with_quantifier_error():
#     bt = bool_type()
#     tq = TypeQuantifier(TypeParam("id1", ir.Kind.Type), bt)
#     try:
#         unify_types(bt, tq)
#         assert False
#     except:
#         return


def test_unify_typevars_with_each_other():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    v3 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    unifier.insert(v2)
    unifier.insert(v3)
    unified = unifier.unify(v1, v2)
    assert (unified == v1 or unified == v2)
    assert unified != v3
    new_unified = unifier.unify(v1, v3)
    assert (new_unified == v1 or new_unified == v2 or new_unified == v3)


def test_unify_typevars_with_basetype():
    unifier = mk.TypeUnifier()
    bt = bool_type()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    unifier.insert(v2)
    unified1 = unifier.unify(v1, bt)
    assert unified1 == bt
    unified2 = unifier.unify(v1, v2)
    assert unified2 == bt


def test_unify_compatible_typevars():
    unifier = mk.TypeUnifier()
    bt = bool_type()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    unifier.insert(v2)
    unifier.unify(v1, bt)
    unifier.unify(v2, bt)
    # because types to which v1 and v2 have been assigned are compatible,
    # this should proceed without problems
    unified = unifier.unify(v1, v2)
    assert unified == bt

# def test_unify_incompatible_typevars():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.Type)
#     v2 = mk.IncompleteType(ir.Kind.Type)
#     bt = bool_type()
#     tq = TypeQuantifier(TypeParam("id1", ir.Kind.Type), bt)
#     unifier.insert(v1)
#     unifier.insert(v2)
#     unifier.unify(v1, bt)
#     unifier.unify(v2, tq)
#     # bt cannot be unified with tq, so unifying v1 and v2 should give an error
#     try:
#         unifier.unify(v1, v2)
#         assert False
#     except:
#         return

# def test_unify_typevar_with_quantifier():
#     unifier = mk.TypeUnifier()
#     tq = TypeQuantifier(TypeParam("id1", ir.Kind.Type), bool_type())
#     v1 = mk.IncompleteType(ir.Kind.BaseType)
#     unifier.insert(v1)
#     unified = unifier.unify(v1, tq)
#     assert unified == tq

# def test_unify_typevars_inside_concrete_quantifier():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.BaseType)
#     unifier.insert(v1)
#     tq1 = TypeQuantifier(TypeParam("id1", ir.Kind.Type), v1)
#     tq2 = TypeQuantifier(TypeParam("id2", ir.Kind.Type), bool_type())
#     unified = unifier.unify(tq1, tq2)
#     assert unified == tq2


def test_unify_concrete_tensors():
    bt = build.bool_dtype()
    shape = tvm.convert([1, 2, 3])
    tt1 = mk.TensorType(shape, bt)
    tt2 = mk.TensorType(shape, bt)
    unified = unify_types(tt1, tt2)
    assert unified == tt1


def test_unify_tensor_shape_reject():
    bt = build.bool_dtype()
    shape1 = tvm.convert([1, 2, 3])
    shape2 = tvm.convert([2, 3, 4])
    tt1 = mk.TensorType(shape1, bt)
    tt2 = mk.TensorType(shape2, bt)
    try:
        unify_types(tt1, tt2)
        assert False
    except:
        return


def test_unify_tensor_dtype_reject():
    bt1 = build.bool_dtype()
    bt2 = build.int_dtype()
    shape = tvm.convert([1, 2, 3])
    tt1 = mk.TensorType(shape, bt1)
    tt2 = mk.TensorType(shape, bt2)
    try:
        unify_types(tt1, tt2)
        assert False
    except:
        return

# def test_unify_quantified_tensors():
#     x = TypeParam("x", ir.type.Kind.Shape)
#     y = TypeParam("y", ir.type.Kind.Shape)
#     tq1 = TypeQuantifier(x, mk.TensorType(bool_type(), x))
#     tq2 = TypeQuantifier(y, mk.TensorType(bool_type(), y))
#     unified = unify_types(tq1, tq2)
#     assert unified == tq1

#     a = TypeParam("a", ir.type.Kind.BaseType)
#     b = TypeParam("b", ir.type.Kind.BaseType)
#     tq3 = TypeQuantifier(a, mk.TensorType(a, make_shape([1, 2, 3])))
#     tq4 = TypeQuantifier(b, mk.TensorType(b, make_shape([1, 2, 3])))
#     unified = unify_types(tq3, tq4)
#     assert unified == tq3

# def test_unify_concrete_products():
#     bt = bool_type()
#     intty = int_type()
#     pt1 = TupleType([bt, intty])
#     pt2 = TupleType([bt, intty])
#     unified = unify_types(pt1, pt2)
#     assert unified == pt1

# def test_unify_products_reject_size():
#     bt = bool_type()
#     intty = IntType(32)
#     pt1 = TupleType([bt, bt, intty])
#     pt2 = TupleType([bt, intty])
#     try:
#         unify_types(pt1, pt2)
#         assert False
#     except:
#         return

# def test_unify_products_reject_member():
#     bt = bool_type()
#     intty = int_type()
#     pt1 = TupleType([bt, bt])
#     pt2 = TupleType([bt, intty])
#     try:
#         unify_types(pt1, pt2)
#         assert False
#     except:
#         return

# def test_unify_products_typevar():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.BaseType)
#     bt = bool_type()
#     pt1 = TupleType([bt, bt])
#     pt2 = TupleType([v1, bt])
#     unifier.insert(v1)
#     unified = unifier.unify(pt1, pt2)
#     assert unified == pt1

# def test_unify_quantified_products():
#     x = TypeParam("x", ir.Kind.Type)
#     y = TypeParam("y", ir.Kind.Type)
#     p1 = TypeQuantifier(x, TupleType([int_type(), x]))
#     p2 = TypeQuantifier(y, TupleType([int_type(), y]))
#     unified = unify_types(p1, p2)
#     assert unified == p1


def test_subst_basetype():
    unifier = mk.TypeUnifier()
    bt = bool_type()
    assert bt == unifier.subst(bt)


def test_subst_simple_hole():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.BaseType)
    bt = bool_type()
    unifier.insert(v1)
    unifier.unify(v1, bt)
    assert unifier.subst(v1) == bt


def test_subst_typevar_for_typevar():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    unifier.insert(v2)

    unifier.unify(v1, v2)
    assert unifier.subst(v1) == unifier.subst(v2)


def test_subst_typevar_for_typevar_comm():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.Type)
    v2 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    unifier.insert(v2)

    unifier.unify(v2, v1)
    assert unifier.subst(v1) == unifier.subst(v2)


def test_subst_concrete_arrow():
    unifier = mk.TypeUnifier()
    arr1 = func_type([int_type()], int_type())
    assert unifier.subst(arr1) == arr1


def test_subst_arrow_with_holes():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.BaseType)
    v2 = mk.IncompleteType(ir.Kind.BaseType)
    unifier.insert(v1)
    unifier.insert(v2)
    unifier.unify(v1, int_type())
    unifier.unify(v2, bool_type())
    arr1 = func_type([v1], v2)
    arr2 = func_type([int_type()], bool_type())
    assert unifier.subst(arr1) == arr2

# def test_subst_concrete_quantifier():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.BaseType)
#     tq = TypeQuantifier(TypeParam("id1", ir.Kind.Type), int_type())
#     unifier.insert(v1)
#     unifier.unify(v1, tq)
#     assert unifier.subst(v1) == tq

# def test_subst_quantifier_with_holes():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.Type)
#     v2 = mk.IncompleteType(ir.Kind.Type)
#     tq1 = TypeQuantifier(TypeParam("id1", ir.Kind.Type), v2)
#     intty = int_type()
#     tq2 = TypeQuantifier(TypeParam("id2", ir.Kind.Type), intty)
    # unifier.insert(v1)
    # unifier.insert(v2)
    # unifier.unify(v2, intty)
    # unifier.unify(v1, tq1)
    # assert unifier.subst(v1) == tq2


def test_subst_concrete_tensor():
    unifier = mk.TypeUnifier()
    v1 = mk.IncompleteType(ir.Kind.Type)
    unifier.insert(v1)
    tt = mk.TensorType(tvm.convert([1, 2, 3]), 'uint1')
    unifier.unify(v1, tt)
    assert unifier.subst(v1) == tt

# def test_subst_concrete_product():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.Type)
#     unifier.insert(v1)
#     bt = bool_type()
#     pt = TupleType([bt, bt])
#     unifier.unify(v1, pt)
#     assert unifier.subst(v1) == pt

# def test_subst_product_with_holes():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.Type)
#     v2 = mk.IncompleteType(ir.Kind.Type)
#     v3 = mk.IncompleteType(ir.Kind.Type)
#     unifier.insert(v1)
#     unifier.insert(v2)
#     unifier.insert(v3)

#     tt1 = mk.TensorType(int_type(), tvm.convert([]))
#     tt2 = mk.TensorType(FloatType(32), tvm.convert([]))
#     pt1 = TupleType([tt1, v2, v3])
#     unifier.unify(v2, tt2)
#     unifier.unify(v3, v2)
#     unifier.unify(v1, pt1)
#     pt2 = TupleType([tt1, tt2, tt2])
#     assert unifier.subst(v1) == pt2

# def test_subst_concrete_ref():
#     unifier = mk.TypeUnifier()
#     rt = RefType(bool_type())
#     assert unifier.subst(rt) == rt

# def test_subst_ref_with_hole():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.Type)
#     unifier.insert(v1)

#     unifier.unify(v1, bool_type())
#     rt1 = RefType(v1)
#     rt2 = RefType(bool_type())
#     assert unifier.subst(rt1) == rt2

# def test_typevar_on_lhs():
#     unifier = mk.TypeUnifier()
#     v1 = mk.IncompleteType(ir.Kind.BaseType)
#     v2 = mk.IncompleteType(ir.Kind.Type)
#     bt = bool_type()
#     tq = TypeQuantifier(TypeParam("id1", ir.Kind.Type), bt, bt)
#     unifier.insert(v1)
#     unifier.insert(v2)
#     unified1 = unifier.unify(bt, v1)
#     assert unified1 == bt
#     unified2 = unifier.unify(tq, v2)
#     assert unified2 == tq
#     assert unifier.subst(v1) == bt
#     assert unifier.subst(v2) == tq