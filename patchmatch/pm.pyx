from libc.stdint cimport int32_t
from libc.math cimport sqrt, ceil, exp, fabs
import numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX

init='HI'

cdef float (*cy_match_fn)(
        float[:, :, ::1],
        (int, int),
        float[:, :, ::1],
        (int, int),
        int,
        float) nogil

cdef float HUGE = 1e25

cy_match_fn=cy_match_l1

def set_match_metric(metric):
    global cy_match_fn
    if metric == 'l1':
        cy_match_fn = cy_match_l1
    elif metric == 'l2': 
        cy_match_fn = cy_match_l2
    elif metric == 'cos':
        cy_match_fn = cy_match_cos
    else:
        print('metric none of l1, l2 or cos')
        assert False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_match_l1(
        float[:, :, ::1] a,
        (int, int) a_pos,
        float[:, :, ::1] b,
        (int, int) b_pos,
        int ks,
        float thresh) nogil:
    cdef float total = 0
    cdef float diff
    cdef int i, j
    cdef int z = a.shape[2]
    cdef int kz = ks * z
    cdef float* a_ptr
    cdef float* b_ptr
    for i in range(ks):
        a_ptr = &a[a_pos[0]+i, a_pos[1], 0]
        b_ptr = &b[b_pos[0]+i, b_pos[1], 0]
        for j in range(kz):
            diff = a_ptr[j] - b_ptr[j]
            total += fabs(diff)
            if total > thresh:
                return total
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_match_l2(
        float[:, :, ::1] a,
        (int, int) a_pos,
        float[:, :, ::1] b,
        (int, int) b_pos,
        int ks,
        float thresh) nogil:
    cdef float total = 0
    cdef float diff
    cdef int i, j
    cdef int z = a.shape[2]
    cdef int kz = ks * z
    cdef float* a_ptr
    cdef float* b_ptr
    for i in range(ks):
        a_ptr = &a[a_pos[0]+i, a_pos[1], 0]
        b_ptr = &b[b_pos[0]+i, b_pos[1], 0]
        for j in range(kz):
            diff = a_ptr[j] - b_ptr[j]
            total += diff * diff
            if total > thresh:
                return total
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cy_match_cos(
        float[:, :, ::1] a,
        (int, int) a_pos,
        float[:, :, ::1] b,
        (int, int) b_pos,
        int ks,
        float thresh) nogil:
    cdef float total = 0
    cdef float diff
    cdef int i, j
    cdef int z = a.shape[2]
    cdef int kz = ks * z
    cdef float* a_ptr
    cdef float* b_ptr
    cdef float la = 0
    cdef float lb = 0
    for i in range(ks):
        a_ptr = &a[a_pos[0]+i, a_pos[1], 0]
        b_ptr = &b[b_pos[0]+i, b_pos[1], 0]
        for j in range(kz):
            la += a_ptr[j] * a_ptr[j]
            lb += b_ptr[j] * b_ptr[j]
            diff = a_ptr[j] * b_ptr[j]
            total += diff * diff
    return -total / (sqrt(la) * sqrt(lb))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) get_loc(int[:, :, ::1] src, (int, int) l) nogil:
    return (src[l[0], l[1], 0], src[l[0], l[1], 1])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) set_loc(
        int[:, :, ::1] src,
        (int, int) l,
        (int, int) x) nogil:
    src[l[0], l[1], 0] = x[0]
    src[l[0], l[1], 1] = x[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) loc_add((int, int) l, (int, int) v) nogil:
    return (l[0] + v[0], l[1] + v[1])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline (int, int) bounds((int, int) l, float[:, :, :]  arr, int k) nogil:
    return (max(0, min(l[0], arr.shape[0] - k)),
            max(0, min(l[1], arr.shape[1] - k)))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float cy_propagate(
        int[:, :, ::1] sub,
        (int, int) ij,
        float[:, :, ::1] src,
        float[:, :, ::1] tgt,
        int k,
        int reverse) nogil:
    cdef int i = ij[0]
    cdef int j = ij[1]
    cdef float this_cost = cy_match_fn(src, ij, tgt, get_loc(sub, ij), k, HUGE)

    cdef (int, int) candidate
    cdef float test_cost
    cdef float[:, :, :] that_patch

    if reverse == 0:
        if i - 1 >= 0:
            candidate = bounds(loc_add(get_loc(sub, (i - 1, j)), (1, 0)), tgt, k)
            test_cost = cy_match_fn(src, ij, tgt, candidate, k, this_cost)
            if test_cost < this_cost:
                set_loc(sub, ij, candidate)
                this_cost = test_cost

        if j - 1 >= 0:
            candidate = bounds(loc_add(get_loc(sub, (i, j - 1)), (0, 1)), tgt, k)
            test_cost = cy_match_fn(src, ij, tgt, candidate, k, this_cost)
            if test_cost < this_cost:
                set_loc(sub, ij, candidate)
                this_cost = test_cost
    else:
        if i + 1 < sub.shape[0]:
            candidate = bounds(loc_add(get_loc(sub, (i + 1, j)), (-1, 0)), tgt, k)
            test_cost = cy_match_fn(src, ij, tgt, candidate, k, this_cost)
            if test_cost < this_cost:
                set_loc(sub, ij, candidate)
                this_cost = test_cost

        if j + 1 < sub.shape[1]:
            candidate = bounds(loc_add(get_loc(sub, (i, j + 1)),  (0, -1)), tgt, k)
            test_cost = cy_match_fn(src, ij, tgt, candidate, k, this_cost)
            if test_cost < this_cost:
                set_loc(sub, ij, candidate)
                this_cost = test_cost
    return this_cost


cdef int continue_RS((float, float) l) nogil:
    return l[0] > 1 or l[1] > 1


@cython.cdivision(True)
cdef inline float rnd_uni() nogil:
    return (<float>rand() / RAND_MAX * 2) - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float cy_random_search(
        int[:, :, ::1] sub,
        (int, int) ij,
        float[:, :, ::1] src,
        float[:, :, ::1] tgt,
        int k) nogil:
    cdef int i = ij[0]
    cdef int j = ij[1]
    cdef (int, int) v = get_loc(sub, ij)
    cdef (int, int) best = v
    cdef float this_cost = cy_match_fn(src, ij, tgt, v, k, HUGE)
    cdef (int, int) total_size = (tgt.shape[0], tgt.shape[1])
    cdef (float, float) r = (total_size[0] * rnd_uni(), total_size[1] * rnd_uni())
    cdef (int, int) u
    cdef float candidate_cost
    while continue_RS(r):
        u[0] = v[0] + <int>r[0]
        u[1] = v[1] + <int>r[1]
        u = bounds(u, tgt, k)
        candidate_cost = cy_match_fn(src, ij, tgt, u, k, this_cost)
        if candidate_cost < this_cost:
            best = u
            this_cost = candidate_cost
        r[0] = r[0] / <float>2.0
        r[1] = r[1] / <float>2.0
    set_loc(sub, ij, best)
    return this_cost

@cython.cdivision(True)
def cy_PatchMatch(
        src,
        tgt,
        int k,
        int n_it=10,
        int min_n_it=4,
        nn=None,
        float thresh_factor=0.99):
    assert min_n_it < n_it
    if nn is None:
        nn = np.stack([
            np.random.randint(
                tgt.shape[0],
                size=(<int>ceil(src.shape[0]), <int>ceil(src.shape[1])),
                dtype='int32'),
            np.random.randint(
                tgt.shape[1],
                size=(<int>ceil(src.shape[0]), <int>ceil(src.shape[1])),
                dtype='int32')
        ], axis=2)
    else:
        assert nn.shape[0] == src.shape[0] and nn.shape[1] == src.shape[1]
        nn[:, :, 0] = nn[:, :, 0].clip(0, tgt.shape[0]-1)
        nn[:, :, 1] = nn[:, :, 1].clip(0, tgt.shape[1]-1)

    cdef int[:, :, ::1] sub = nn
    cdef int m = k >> 1
    cdef float[:, :, ::1] src_yuv = np.pad(
            src, ((m, m), (m, m), (0, 0)), 'reflect')
    cdef float[:, :, ::1] tgt_yuv = np.pad(
            tgt, ((m, m), (m, m), (0, 0)), 'reflect')
    cdef int e
    cdef int i, j
    cdef float total = 0
    cdef float total_row = 0
    cdef float prev_total = 1e25
    for e in range(n_it):
        total = 0
        if e & 1 == 0:
            for i in range(sub.shape[0]):
                total_row = 0
                for j in range(sub.shape[1]):
                    cy_propagate(sub, (i, j), src_yuv, tgt_yuv, k, False)
                    total_row += cy_random_search(sub, (i, j), src_yuv, tgt_yuv, k)
                total += total_row
        else:
            for i in reversed(range(sub.shape[0])):
                total_row = 0
                for j in reversed(range(sub.shape[1])):
                    cy_propagate(sub, (i, j), src_yuv, tgt_yuv, k, True)
                    total_row += cy_random_search(sub, (i, j), src_yuv, tgt_yuv, k)
                total += total_row

        if total / (prev_total + 1e-8) > thresh_factor and e >= min_n_it:
            break
        prev_total = total
    return nn, total


def make_vote_arena(size):
    return np.zeros((size[0], size[1], size[2] + 1), dtype=np.float32)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def render_arena(arena):
    cdef int z = arena.shape[2] - 1
    cdef float[:, :, ::1] res_v = arena
    cdef int i
    cdef int j
    cdef int k
    for i in range(res_v.shape[0]):
        for j in range(res_v.shape[1]):
            for k in range(res_v.shape[2] - 1):
                res_v[i, j, k] = res_v[i, j, k] / res_v[i, j, z]
    return np.ascontiguousarray(arena[:, :, :z])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def vote(idx, arena, tgt, int k):
    original_shape = idx.shape

    cdef int[:, :, ::1] idx_v = idx
    cdef (int, int) patch
    cdef int m = k//2
    cdef int i
    cdef int j
    cdef int pi
    cdef int pj
    cdef int pk
    cdef float p
    cdef float ns = idx.shape[0] * idx.shape[1]
    cdef float[:, :, ::1] tgt_v = np.pad(tgt, ((m, m), (m, m), (0, 0)), 'reflect')
    res = np.pad(arena, ((m, m), (m, m), (0, 0)), mode='reflect')

    cdef float[:, :, ::1] res_v = res
    cdef int z = res_v.shape[2] - 1
    cdef float* res_vi

    for i in range(idx_v.shape[0]):
        for j in range(idx_v.shape[1]):
            patch = get_loc(idx_v, (i, j))
            for pi in range(k):
                res_vi = &res_v[i + pi, 0, 0]
                for pj in range(k):
                    for pk in range(z):
                        p = tgt_v[patch[0] + pi, patch[1] + pj, pk]
                        res_v[i + pi, j + pj, pk] += p * ns
                    res_v[i + pi, j + pj, z] += ns

    return np.ascontiguousarray(
            res[m:m + original_shape[0], m:m + original_shape[1], :])

@cython.cdivision(True)
@cython.boundscheck(False)
#@cython.wraparound(False)
def reverse_vote(idx, arena, tgt, int k):
    original_shape = arena.shape

    cdef int[:, :, ::1] idx_v = idx
    cdef (int, int) patch
    cdef int m = k//2
    cdef int i
    cdef int j
    cdef int pi
    cdef int pj
    cdef int pk
    cdef float p
    cdef float[:, :, ::1] tgt_v = np.pad(
            tgt, ((m, m), (m, m), (0, 0)), 'reflect')
    res = np.pad(arena, ((m, m), (m, m), (0, 0)), mode='reflect')

    cdef float[:, :, ::1] res_v = res
    cdef int z = res_v.shape[2] - 1
    cdef (int, int) dst

    for i in range(idx_v.shape[0]):
        for j in range(idx_v.shape[1]):
            dst = get_loc(idx_v, (i, j))
            for pi in range(k):
                for pj in range(k):
                    for pk in range(z):
                        p = tgt_v[i + pi, j + pj, pk]
                        res_v[dst[0] + pi, dst[1] + pj, pk] += p
                    res_v[dst[0] + pi, dst[1] + pj, z] += 1

    return np.ascontiguousarray(
            res[m:m + original_shape[0], m:m + original_shape[1], :])
