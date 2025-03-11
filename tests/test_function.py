import pytest
from types import SimpleNamespace
from vyper import ast as vy_ast
from vyper.semantics.types.function import ContractFunctionT, MemberFunctionT, PositionalArg, KeywordArg
from vyper.semantics.analysis.base import FunctionVisibility, StateMutability, VarOffset

###########################################################################
# Dummy/fake types and helper functions for testing purposes.
###########################################################################
class DummyType:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def compare_type(self, other):
        return isinstance(other, DummyType) and self.name == other.name
    @property
    def canonical_abi_type(self):
        return self.name
    def selector_name(self):
        return self.name
    @property
    def abi_type(self):
        return self
    def to_abi_arg(self, name=""):
        return {"name": name, "type": self.name}

# Dummy AST node to simulate Vyper AST nodes.
class DummyAST(SimpleNamespace):
    pass

def create_dummy_call(arguments):
    """
    Create a dummy call node for testing fetch_call_return which expects:
        - a func attribute with a value that has _supports_external_calls = True.
        - an args attribute containing the provided arguments.
        - a keywords attribute and node_source_code attribute.
    """
    dummy_value = DummyAST(_supports_external_calls=True, _metadata={"dummy": None})
    dummy_attr = vy_ast.Attribute(value=dummy_value, attr="dummy")
    return SimpleNamespace(
        func=dummy_attr,
        args=arguments,
        keywords=[],
        node_source_code="dummy_call_code"
    )

# Dummy validation functions to bypass actual type checking.
def dummy_validate_call_args(node, expected, kwarg_keys=None):
    pass

def dummy_validate_expected_type(arg, expected_type):
    pass

# Monkey-patch the validation functions in the tested module.
import vyper.semantics.types.function as vf
vf.validate_call_args = dummy_validate_call_args
vf.validate_expected_type = dummy_validate_expected_type
vf.get_exact_type_from_node = lambda node: node

###########################################################################
# Test cases for ContractFunctionT and MemberFunctionT.
###########################################################################

def test_contract_function_str_and_repr():
    """Test __str__ and __repr__ methods of ContractFunctionT."""
    pos_arg = PositionalArg("a", DummyType("int256"))
    kw_arg = KeywordArg(name="b", typ=DummyType("uint256"), default_value=DummyAST())
    func = ContractFunctionT(
        name="foo",
        positional_args=[pos_arg],
        keyword_args=[kw_arg],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    s = str(func)
    r = repr(func)
    assert "foo(" in s
    assert "->" in s
    assert "contract function foo(" in r

def test_method_ids_without_defaults():
    """Test method_ids property for a function without default arguments."""
    pos_arg = PositionalArg("x", DummyType("int256"))
    func = ContractFunctionT(
        name="bar",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=None,
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.VIEW,
        nonreentrant=False
    )
    m_ids = func.method_ids
    # method_ids should contain one entry
    assert isinstance(m_ids, dict)
    assert len(m_ids) == 1
    for sig in m_ids.keys():
        assert sig.startswith("bar(")

def test_method_ids_with_defaults():
    """Test method_ids property for a function with default arguments."""
    pos_arg = PositionalArg("x", DummyType("int256"))
    kw_arg = KeywordArg(name="y", typ=DummyType("uint256"), default_value=DummyAST())
    func = ContractFunctionT(
        name="baz",
        positional_args=[pos_arg],
        keyword_args=[kw_arg],
        return_type=DummyType("int256"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    m_ids = func.method_ids
    # method_ids should have two entries: one with just positional and one with positional+keyword.
    assert len(m_ids) == 2
    sigs = list(m_ids.keys())
    assert "baz(int256)" in sigs
    assert "baz(int256,uint256)" in sigs

def test_fetch_call_return_success():
    """Test fetch_call_return successfully validates a call."""
    pos_arg = PositionalArg("x", DummyType("int256"))
    func = ContractFunctionT(
        name="qux",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    dummy_arg = DummyAST()
    dummy_arg.node_source_code = "dummy_arg"
    call_node = create_dummy_call([dummy_arg])
    ret_type = func.fetch_call_return(call_node)
    assert ret_type == func.return_type

def test_getter_from_variable_decl():
    """Test getter_from_VariableDecl method of ContractFunctionT."""
    dummy_ast = SimpleNamespace(id="var", is_public=True)
    dummy_ast.target = SimpleNamespace(
        id="var",
        _metadata={"varinfo": SimpleNamespace(typ=DummyType("int256"))}
    )
    # Initially, getter_signature might not be defined on DummyType.
    # Set it for testing purposes.
    DummyType.getter_signature = ([DummyType("int256")], DummyType("int256"))
    getter_func = ContractFunctionT.getter_from_VariableDecl(dummy_ast)
    assert getter_func.name == "var"
    assert getter_func.visibility == FunctionVisibility.EXTERNAL
    assert getter_func.n_positional_args == 1

def test_member_function_fetch_call_return():
    """Test fetch_call_return of MemberFunctionT."""
    dummy_arg = DummyAST()
    dummy_arg.node_source_code = "dummy_arg"
    call_node = SimpleNamespace(
        args=[dummy_arg],
        node_source_code="dummy_member_call"
    )
    member_func = MemberFunctionT(
        underlying_type=DummyType("DynArray[int128,3]"),
        name="append",
        arg_types=[DummyType("int128")],
        return_type=None,
        is_modifying=True
    )
    ret = member_func.fetch_call_return(call_node)
    assert ret is None

def test_abi_signature_for_kwargs():
    """Test the abi_signature_for_kwargs method of ContractFunctionT."""
    pos_arg = PositionalArg("a", DummyType("int"))
    kw_arg = KeywordArg(name="b", typ=DummyType("uint"), default_value=DummyAST())
    func = ContractFunctionT(
        name="sigtest",
        positional_args=[pos_arg],
        keyword_args=[kw_arg],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PURE,
        nonreentrant=False
    )
    sig = func.abi_signature_for_kwargs([kw_arg])
    assert sig == "sigtest(int,uint)"
def test_fetch_call_return_non_payable():
    """Test that fetch_call_return raises CallViolation for nonpayable functions when a 'value' keyword is provided."""
    from vyper.exceptions import CallViolation
    pos_arg = PositionalArg("x", DummyType("int256"))
    func = ContractFunctionT(
        name="nonpayable_func",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.NONPAYABLE,
        nonreentrant=False
    )
    dummy_arg = DummyAST()
    dummy_arg.node_source_code = "dummy_arg"
    # Create a dummy keyword node representing value passed
    dummy_kwarg = DummyAST(arg="value")
    dummy_kwarg.node_source_code = "value=100"
    call_node = SimpleNamespace(
        func=create_dummy_call([]).func,
        args=[dummy_arg],
        keywords=[SimpleNamespace(arg="value", value=dummy_kwarg)],
        node_source_code="nonpayable_call"
    )
    with pytest.raises(CallViolation):
        func.fetch_call_return(call_node)

def test_fetch_call_return_unexpected_kwarg():
    """Test that fetch_call_return raises ArgumentException for an unexpected keyword argument."""
    from vyper.exceptions import ArgumentException
    pos_arg = PositionalArg("x", DummyType("int256"))
    func = ContractFunctionT(
        name="unexpected_kwarg_func",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    dummy_arg = DummyAST()
    dummy_arg.node_source_code = "dummy_arg"
    # Provide an unexpected keyword argument 'unexpected'
    dummy_kwarg = DummyAST(arg="unexpected")
    dummy_kwarg.node_source_code = "unexpected=123"
    call_node = SimpleNamespace(
        func=create_dummy_call([]).func,
        args=[dummy_arg],
        keywords=[SimpleNamespace(arg="unexpected", value=dummy_kwarg)],
        node_source_code="unexpected_kwarg_call"
    )
    with pytest.raises(ArgumentException):
        func.fetch_call_return(call_node)

def test_set_reentrancy_key_position_errors():
    """Test set_reentrancy_key_position method error conditions for ContractFunctionT."""
    from vyper.exceptions import CompilerPanic
    # Test that a non-nonreentrant function raises CompilerPanic when setting a key position.
    pos_arg = PositionalArg("x", DummyType("int256"))
    nonreentrant_func = ContractFunctionT(
        name="non_reentrant",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    with pytest.raises(CompilerPanic):
        nonreentrant_func.set_reentrancy_key_position(VarOffset(0))

    # Test that reassigning an already-set key position raises CompilerPanic.
    reentrant_func = ContractFunctionT(
        name="reentrant",
        positional_args=[pos_arg],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=True
    )
    # First assignment should succeed.
    reentrant_func.set_reentrancy_key_position(VarOffset(1))
    with pytest.raises(CompilerPanic):
        reentrant_func.set_reentrancy_key_position(VarOffset(2))

def test_implements_method():
    """Test the implements method for ContractFunctionT with matching and non‐matching signatures."""
    pos_arg1 = PositionalArg("a", DummyType("int256"))
    func1 = ContractFunctionT(
        name="impl",
        positional_args=[pos_arg1],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    pos_arg2 = PositionalArg("a", DummyType("int256"))
    func2 = ContractFunctionT(
        name="impl",
        positional_args=[pos_arg2],
        keyword_args=[],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    # Functions with the same signature should implement each other.
    assert func1.implements(func2)

    # Change the return type, should not implement.
    func3 = ContractFunctionT(
        name="impl",
        positional_args=[pos_arg2],
        keyword_args=[],
        return_type=DummyType("int256"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    assert not func1.implements(func3)

def test_member_function_fetch_call_return_invalid_args():
    """Test that MemberFunctionT.fetch_call_return raises an error when provided an incorrect number of arguments."""
    from vyper.exceptions import ArgumentException
    member_func = MemberFunctionT(
        underlying_type=DummyType("DynArray[int128,3]"),
        name="append",
        arg_types=[DummyType("int128")],
        return_type=None,
        is_modifying=True
    )
    # Create a call node with no arguments (should trigger the assert)
    call_node = SimpleNamespace(
        args=[],
        node_source_code="dummy_member_call_invalid"
    )
    with pytest.raises(AssertionError):
        member_func.fetch_call_return(call_node)

def test_to_toplevel_abi_dict():
    """Test the to_toplevel_abi_dict method for fallback, constructor, and normal functions."""
    # Test fallback function (default function)
    func_default = ContractFunctionT(
        name="__default__",
        positional_args=[],
        keyword_args=[],
        return_type=None,
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.NONPAYABLE,
        nonreentrant=False
    )
    abi_default = func_default.to_toplevel_abi_dict()
    assert isinstance(abi_default, list)
    assert abi_default[0]["type"] == "fallback"

    # Test constructor function
    func_constructor = ContractFunctionT(
        name="__init__",
        positional_args=[],
        keyword_args=[],
        return_type=None,
        function_visibility=FunctionVisibility.DEPLOY,
        state_mutability=StateMutability.NONPAYABLE,
        nonreentrant=False
    )
    abi_constructor = func_constructor.to_toplevel_abi_dict()
    assert isinstance(abi_constructor, list)
    assert abi_constructor[0]["type"] == "constructor"

    # Test normal external function with a keyword argument (multiple ABI outputs)
    pos_arg = PositionalArg("a", DummyType("int256"))
    kw_arg = KeywordArg(name="b", typ=DummyType("uint256"), default_value=DummyAST())
    func_extra = ContractFunctionT(
        name="normal",
        positional_args=[pos_arg],
        keyword_args=[kw_arg],
        return_type=DummyType("bool"),
        function_visibility=FunctionVisibility.EXTERNAL,
        state_mutability=StateMutability.PAYABLE,
        nonreentrant=False
    )
    abi_extra = func_extra.to_toplevel_abi_dict()
    assert isinstance(abi_extra, list)
    # Should have two entries due to the default keyword argument.
    assert len(abi_extra) == 2