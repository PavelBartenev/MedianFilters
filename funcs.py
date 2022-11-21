import time
import numpy as np
from tqdm import tqdm

pixel_vals = 256


def check_equality(imgs):
    for i in range(1, len(imgs)):
        if (imgs[i] != imgs[i - 1]).any():
            print('Image {} is not equal to image {}'.format(i-1, i))
            return

    print('All images are equal')


def calculate_median(hist, elems_num):
    count_cur = 0
    center_left_elem = None

    if elems_num % 2 == 0:
        for pixel_val in range(pixel_vals):
            if hist[pixel_val] != 0:
                count_cur += hist[pixel_val]
                if count_cur > (elems_num / 2):
                    if not center_left_elem:
                        median = pixel_val
                    else:
                        median = (pixel_val + center_left_elem) // 2

                    return median

                if count_cur == elems_num / 2:
                    center_left_elem = pixel_val
    else:
        for pixel_val in range(pixel_vals):
            count_cur += hist[pixel_val]
            if count_cur >= elems_num // 2 + 1:
                median = pixel_val

                return median


def speed_bechmark(image, filter):
    result = []
    median_filter = filter(image)
    for R in tqdm(range(3, 100, 5)):
        start_time = time.time()
        median_filter.process(R)
        end_time = time.time()

        result.append(1e9 * (end_time - start_time) / (image.shape[0] * image.shape[1]))

    return result


