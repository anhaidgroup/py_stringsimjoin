# Simplified knockoff of nose.tools.raises
# Thanks to zware for writing this for py_stringmatching
def raises(exc_type):
    def deco(f):
        def raises_wrapper(self):
            with self.assertRaises(exc_type):
                return f(self)
        return raises_wrapper
    return deco

# Replacement for nose.tools.nottest
# I have no idea what I'm doing
# At this point this is literally nose.tools.nottest
def nottest(func):
    func.__test__ = False
    return func
