from cv2_process import upsample


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    R = pyr[nlev-1]
    for l in range(nlev-2, -1, -1):
        R = pyr[l] + upsample(R)

    return R