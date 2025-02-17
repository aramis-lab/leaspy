from leaspy.utils.subtypes import suffixed_method
from tests.utils import LeaspyTestCase


class MockClass:
    name = "mock_name"
    _subtype_suffix = "_one"

    @suffixed_method
    def get(self):
        pass

    def get_one(self):
        return 1


class TestSuffixMethod(LeaspyTestCase):
    def test_suffix_ok(self):
        m = MockClass()
        self.assertEqual(m.get(), 1)

    def test_bad_suffix(self):
        m = MockClass()

        m._subtype_suffix = "_two"
        with self.assertRaises(NotImplementedError):
            m.get()

        m._subtype_suffix = None
        with self.assertRaises(AssertionError):
            m.get()
