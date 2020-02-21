import sys
DEFAULT_GROUP = 'kreshuk'
DEFAULT_SHEBANG = sys.executable
DEFAULT_BLOCK_SHAPE = [50, 512, 512]


#
# default group parameter
#

def set_default_group(group):
    global DEFAULT_GROUP
    DEFAULT_GROUP = group


def get_default_group():
    return DEFAULT_GROUP


#
# default shebang parameter
#

def set_default_shebang(shebang):
    global DEFAULT_SHEBANG
    DEFAULT_SHEBANG = shebang


def get_default_shebang():
    return DEFAULT_SHEBANG


#
# default block_shape parameter
#

def set_default_block_shape(block_shape):
    global DEFAULT_BLOCK_SHAPE
    DEFAULT_BLOCK_SHAPE = block_shape


def get_default_block_shape():
    return DEFAULT_BLOCK_SHAPE
