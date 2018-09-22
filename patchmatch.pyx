from libc.stdint cimport int32_t, uint8_t
from libc.math cimport sqrt
import numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX

ctypedef fused img_t:
    uint8_t
    float

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline img_t[:, :, :] cy_extract_patch((int, int) l, img_t[:, :, ::1] src, int k):
    return src[l[0]:l[0] + k, l[1]:l[1] + k, :]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_match_corr(float[:, :, :] a, float[:, :, :] b):
    cdef float total = 0
    cdef float a_len = 0
    cdef float b_len = 0
    cdef float x, y
    cdef int i
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                x = a[i, j, k]
                y = b[i, j, k]
                total += x * y
                a_len += x * x
                b_len += y * y
    return -total / (sqrt(a_len) * sqrt(b_len))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cy_match_l2(uint8_t[:, :, :] a, uint8_t[:, :, :] b):
    cdef int total = 0
    cdef int diff
    cdef int i
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                diff = (<int>a[i, j, k]) - b[i, j, k]
                total += diff * diff

    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_match(img_t[:, :, :] a, img_t[:, :, :] b):
    if img_t is uint8_t:
        return <float>cy_match_l2(a, b)
    else:
        return cy_match_corr(a, b)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) get_loc(int[:, :, ::1] src, (int, int) l):
    return (src[l[0], l[1], 0], src[l[0], l[1], 1])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) set_loc(int[:, :, ::1] src, (int, int) l, (int, int) x):
    src[l[0], l[1], 0] = x[0]
    src[l[0], l[1], 1] = x[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) loc_add((int, int) l, (int, int) v):
    return (l[0] + v[0], l[1] + v[1])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) bounds((int, int) l, img_t[:, :, ::1]  arr, int k):
    cdef int m = k >> 1
    return (max(0, min(l[0], arr.shape[0] - k)), max(0, min(l[1], arr.shape[1] - k)))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cy_propagate(int[:, :, ::1] sub, (int, int) l, img_t[:, :, ::1] src,
        img_t[:, :, ::1] tgt, int k, int reverse):
    cdef int i = l[0]
    cdef int j = l[1]
    cdef img_t[:, :, :] this_patch = cy_extract_patch((i, j), src, k)
    cdef float this_cost = cy_match(this_patch, cy_extract_patch(get_loc(sub, (i, j)), tgt, k))
    
    cdef (int, int) candidate
    cdef float test_cost
    
    if reverse == 0:
        if i - 1 >= 0:
            candidate = bounds(loc_add(get_loc(sub, (i - 1, j)), (1, 0)), tgt, k)
            test_cost = cy_match(this_patch, cy_extract_patch(candidate, tgt, k))
            if test_cost < this_cost:
                set_loc(sub, (i, j), candidate)
                this_cost = test_cost

        if j - 1 >= 0:
            candidate = bounds(loc_add(get_loc(sub, (i, j - 1)), (0, 1)), tgt, k)
            test_cost = cy_match(this_patch, cy_extract_patch(candidate, tgt, k))
            if test_cost < this_cost:
                set_loc(sub, (i, j), candidate)
                this_cost = test_cost
    else:
        if i + 1 < sub.shape[0]:
            candidate = bounds(loc_add(get_loc(sub, (i + 1, j)), (-1, 0)), tgt, k)
            test_cost = cy_match(this_patch, cy_extract_patch(candidate, tgt, k))
            if test_cost < this_cost:
                set_loc(sub, (i, j), candidate)
                this_cost = test_cost

        if j + 1 < sub.shape[1]:
            candidate = bounds(loc_add(get_loc(sub, (i, j + 1)),  (0, -1)), tgt, k)
            test_cost = cy_match(this_patch, cy_extract_patch(candidate, tgt, k))
            if test_cost < this_cost:
                set_loc(sub, (i, j), candidate)
                this_cost = test_cost


cdef int continue_RS((float, float) l):
    return l[0] > 1 or l[1] > 1


@cython.cdivision(True)
cdef inline float rnd_uni():
    return (rand() / RAND_MAX * 2) - 1


cdef float ipow(float x, int32_t p):
    cdef float a = 1
    while p > 0:
        if p & 1 == 0:
            x *= x
            p = p >> 1
        else:
            a *= x
            x *= x
            p = p >> 1
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cy_random_search(int[:, :, ::1] sub, (int, int) loc, img_t[:, :, ::1] src, img_t[:, :, ::1] tgt, int k):
    cdef int i = loc[0]
    cdef int j = loc[1]
    cdef img_t[:, :, :] this_patch = cy_extract_patch((i, j), src, k)
    cdef int e = 0
    cdef (int, int) v = get_loc(sub, (i, j))
    cdef float this_cost = cy_match(this_patch, cy_extract_patch(v, tgt, k))
    cdef (int, int) total_size = (tgt.shape[0], tgt.shape[1])
    cdef (float, float) r = (<float>total_size[0], <float>total_size[1])
    cdef (int, int) u
    cdef float candidate_cost
    while continue_RS(r):
        u[0] = <int>(v[0] + r[0] * rnd_uni())
        u[1] = <int>(v[1] + r[1] * rnd_uni())
        u = bounds(u, tgt, k)
        candidate_cost = cy_match(this_patch, cy_extract_patch(u, tgt, k))
        if candidate_cost < this_cost:
            set_loc(sub, (i, j), u)
            this_cost = candidate_cost
        e += 1
        r[0] = total_size[0] * ipow(0.5, e)
        r[1] = total_size[1] * ipow(0.5, e)


cdef cy_patchmatch_loop(int[:, :, ::1] sub, img_t[:, :, ::1] src, img_t[:, :, ::1] tgt, int k,
                       int n_it):
    cdef int i, j
    cdef int e
    for e in range(n_it):
        if e & 1 == 0:
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    cy_propagate(sub, (i, j), src, tgt, k, False)
                    cy_random_search(sub, (i, j), src, tgt, k)
        else:
            for i in reversed(range(sub.shape[0])):
                for j in reversed(range(sub.shape[1])):
                    cy_propagate(sub, (i, j), src, tgt, k, True)
                    cy_random_search(sub, (i, j), src, tgt, k)
    

def cy_PatchMatch(src, tgt, int k, int n_it=5):
    cdef float[:, :, ::1] src_f
    cdef float[:, :, ::1] tgt_f
    cdef uint8_t[:, :, ::1] src_u8
    cdef uint8_t[:, :, ::1] tgt_u8
    nn = np.stack([
        np.random.randint(tgt.shape[0], size=src.shape[:-1], dtype='int32'),
        np.random.randint(tgt.shape[1], size=src.shape[:-1], dtype='int32')
    ], axis=2)
    cdef int[:, :, ::1] nn_v = nn
    cdef int m = k >> 1
    src = np.pad(src, ((m, m), (m, m), (0, 0)), 'constant')
    tgt = np.pad(tgt, ((m, m), (m, m), (0, 0)), 'constant')
    if src.dtype == np.float32:
        src_f = src
        tgt_f = tgt
        cy_patchmatch_loop(nn_v, src_f, tgt_f, k, n_it)
        return nn
    elif src.dtype == np.uint8:
        src_u8 = src
        tgt_u8 = tgt
        cy_patchmatch_loop(nn_v, src_u8, tgt_u8, k, n_it)
        return nn

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef reconstruct_inner(int [:, :, ::1] idx_v, img_t[:, :, ::1] tgt_v, float[:, :, ::1] res_v, int k):
    cdef img_t[:, :, :] patch
    cdef int i
    cdef int j
    cdef int pi
    cdef int pj
    cdef int pk
    cdef int z = res_v.shape[2] - 1

    for i in range(res_v.shape[0] - k + 1):
        for j in range(res_v.shape[1] - k + 1):
            patch = cy_extract_patch(get_loc(idx_v, (i, j)), tgt_v, k)
            for pi in range(k):
                for pj in range(k):
                    for pk in range(z):
                        p = patch[pi, pj, pk]
                        res_v[i + pi, j + pj, pk] += p
                    res_v[i + pi, j + pj, z] += 1

    for i in range(res_v.shape[0]):
        for j in range(res_v.shape[1]):
            for k in range(res_v.shape[2] - 1):
                res_v[i, j, k] = res_v[i, j, k] / res_v[i, j, z]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def reconstruct(int[:, :, ::1] idx, tgt, int k):
    cdef int m = k // 2
    tgt = np.pad(tgt, ((m, m), (m, m), (0, 0)), 'constant')
    res = np.zeros((idx.shape[0] + 2 * m, idx.shape[1] + 2 * m, tgt.shape[2] + 1), dtype='float32')
    cdef int z = res.shape[2] - 1

    cdef uint8_t[:, :, ::1] tgt_u8
    cdef float[:, :, ::1] res_f, tgt_f
    res_f = res
    if tgt.dtype == np.float32:
        tgt_f = tgt
        reconstruct_inner(idx, tgt_f, res_f, k)
        return res[m:res.shape[0] - m, m:res.shape[1] - m, :z]
    elif tgt.dtype == np.uint8:
        tgt_u8 = tgt
        reconstruct_inner(idx, tgt_u8, res_f, k)
        return res[m:res.shape[0] - m, m:res.shape[1] - m, :z].astype('uint8')

    
    
