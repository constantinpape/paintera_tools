import unittest


# for now only test imports
class TestPainteraToolsIntegration(unittest.TestCase):
    def test_imports(self):
        import paintera_tools
        self.assertTrue(hasattr(paintera_tools, 'serialize_from_commit'))
        self.assertTrue(hasattr(paintera_tools, 'serialize_from_project'))


if __name__ == '__main__':
    unittest.main()
