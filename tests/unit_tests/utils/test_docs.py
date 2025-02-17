import doctest
import re

import leaspy.utils.docs as lsp_docs
from leaspy.utils.docs import doc_with, doc_with_super
from tests.utils import LeaspyTestCase


class A:
    @classmethod
    def cm(cls):
        """Class method of A"""
        ...

    @staticmethod
    def sm():
        """Static method of A"""
        ...

    def m(self):
        """Method for a A instance"""
        ...

    @classmethod
    def cm_bis(cls):
        """Class method of A (bis)"""
        ...

    @staticmethod
    def sm_bis():
        """Static method of A (bis)"""
        ...

    def m_bis(self):
        """Method for a A instance (bis)"""
        ...

    def foo(self, x):
        """
        Can't say the word `less` since I'm wordless. Word-
        """
        return x


class B:
    """Class B"""

    def foo(self, y):
        """Foo doc from B"""
        ...

    def bar(self):
        """Bar from B"""
        ...


@doc_with_super(mapping={"A": "C", "B": "C"})
class C(A, B):
    @classmethod
    def cm(cls): ...

    @staticmethod
    def sm(): ...

    def m(self): ...

    def foo(self, x):
        return

    def bar(self):
        return


@doc_with(
    A.foo,
    {"say": "hear", "word": "***", "less": "!?", "I'm": "you're"},
    flags=re.IGNORECASE,
)
def bar():
    return


class TestDocstringUtils(LeaspyTestCase):
    def test_doc_with(self):
        # doc_with
        self.assertEqual(
            bar.__doc__.strip(), "Can't hear the *** `!?` since you're wordless. ***-"
        )

        # doc_with_super
        self.assertEqual(C.__doc__, "Class C")

        self.assertEqual(
            C.foo.__doc__.strip(), "Can't say the word `less` since I'm wordless. Word-"
        )
        self.assertEqual(C.bar.__doc__, """Bar from C""")

        # not replaced methods (because not re-implemented at all)
        self.assertEqual(C.cm.__doc__, "Class method of C")
        self.assertEqual(C.sm.__doc__, "Static method of C")
        self.assertEqual(C.m.__doc__, """Method for a C instance""")

        # not replaced methods (because not re-implemented at all)
        self.assertEqual(C.cm_bis.__doc__, "Class method of A (bis)")
        self.assertEqual(C.sm_bis.__doc__, "Static method of A (bis)")
        self.assertEqual(C.m_bis.__doc__, """Method for a A instance (bis)""")

        # test all examples contained in docstrings of any object of module
        doctest.testmod(lsp_docs)

    def test_behavior_when_signatures_mismatch(self):
        class Super:
            def mismatch_of_signature(x, *, y):
                """Hello"""

        with self.assertRaisesRegex(
            ValueError, "has a different signature than its parent"
        ):

            @doc_with_super(if_other_signature="raise")
            class Child(Super):
                def mismatch_of_signature(x, y): ...

        with self.assertWarnsRegex(
            UserWarning, "has a different signature than its parent"
        ):

            @doc_with_super(if_other_signature="warn")
            class Child(Super):
                def mismatch_of_signature(x, *, y=None): ...

        self.assertEqual(Child.mismatch_of_signature.__doc__, "Hello")

        @doc_with_super(if_other_signature="skip")
        class Child(Super):
            def mismatch_of_signature(x, *, other_name): ...

        self.assertIsNone(Child.mismatch_of_signature.__doc__)

        @doc_with_super(if_other_signature="force")
        class Child(Super):
            def mismatch_of_signature(x, *, other_name): ...

        self.assertEqual(Child.mismatch_of_signature.__doc__, "Hello")
