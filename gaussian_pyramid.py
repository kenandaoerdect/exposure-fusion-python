from cv2_process import downsample


def gaussian_pyramid(I, nlev):
    pyr = [I]
    for l in range(1, nlev):
        I = downsample(I)
        pyr.append(I)
    return pyr