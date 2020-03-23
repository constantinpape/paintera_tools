import sys
DEFAULT_GROUP = 'kreshuk'
DEFAULT_SHEBANG = sys.executable
DEFAULT_BLOCK_SHAPE = [50, 512, 512]
DEFAULT_QOS = 'normal'
DEFAULT_ROI_BEGIN = None
DEFAULT_ROI_END = None


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
# get default roi parameter
#

def get_default_roi():
    return DEFAULT_ROI_BEGIN, DEFAULT_ROI_END


def set_default_roi(roi_begin, roi_end):
    global DEFAULT_ROI_BEGIN, DEFAULT_ROI_END
    DEFAULT_ROI_BEGIN = roi_begin
    DEFAULT_ROI_END = roi_end


#
# default qos parameter
#

def set_default_qos(qos):
    global DEFAULT_QOS
    DEFAULT_QOS = qos


def get_default_qos():
    return DEFAULT_QOS


#
# default block_shape parameter
#

def set_default_block_shape(block_shape):
    global DEFAULT_BLOCK_SHAPE
    DEFAULT_BLOCK_SHAPE = block_shape


def get_default_block_shape():
    return DEFAULT_BLOCK_SHAPE
