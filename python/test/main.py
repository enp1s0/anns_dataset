import anns_dataset as ad
import numpy as np
import sys

def dtype_conv(a):
    if a == ad.u32:
        return 'uint32'
    if a == ad.i32:
        return 'int32'
    if a == ad.u8:
        return 'uint8'
    if a == ad.i8:
        return 'int8'
    if a == ad.f32:
        return 'float32'

def test_get_shape(path: str, d):
    print("# " + sys._getframe().f_code.co_name)
    size, dim = ad.get_shape(path, d)
    print("size =", size)
    print("dim =", dim)

def test_load_store(d):
    print("# " + sys._getframe().f_code.co_name)
    size = 10000
    dim = 100
    ds_A = np.random.rand(size, dim).astype(dtype_conv(d))

    ad.store(np.asarray(ds_A), 'a.vec', ad.FORMAT_VECS)

    ds_B = ad.load('a.vec', d)

    c = ds_A - ds_B
    diff = np.linalg.norm(c)
    print(diff)
    if diff == 0:
        print("PASSED")
    else:
        print("FAILED")

test_load_store(ad.f32)
test_get_shape('a.vec', ad.f32)
