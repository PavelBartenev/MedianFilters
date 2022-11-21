import numpy as np
from funcs import calculate_median, pixel_vals


class SimpleMedianFilter:
    def __init__(self, img):
        self.img = img
        self.processed = None

    def process(self, R):
        H, W = self.img.shape
        self.processed = np.zeros_like(self.img)

        for i in range(H):
            for j in range(W):
                window = self.img[max(0, i - R // 2) : min(H, i + R // 2 + 1),
                                  max(0, j - R // 2) : min(W, j + R // 2 + 1)]

                sorted_window = np.sort(window.flatten())

                if len(sorted_window) % 2 == 1:
                    self.processed[i, j] = sorted_window[len(sorted_window) // 2]
                else:
                    median = (sorted_window[len(sorted_window) // 2 - 1].astype(np.int64)
                            + sorted_window[len(sorted_window) // 2].astype(np.int64)) // 2

                    self.processed[i, j] = median


class HuangMedianFilter:
    def __init__(self, img):
        self.img = img
        self.processed = None
        self.pixel_vals = 256

    def process(self, R):
        H, W = self.img.shape
        self.processed = np.zeros_like(self.img)

        for i in range(H):
            window = self.img[max(0, i - (R // 2)) : min(H, (R // 2) + i + 1),
                              0 : (R // 2) + 1]

            elems_num = window.shape[0] * window.shape[1]
            hist = np.histogram(window.flatten(), bins=[x for x in range(self.pixel_vals + 1)])[0]

            self.processed[i, 0] = calculate_median(hist, elems_num)

            for j in range(1, W):
                for t in range(-(R // 2), R // 2 + 1):
                    if 0 <= i + t < H:
                        if j - (R // 2) - 1 >= 0:
                            x = self.img[i + t, j - (R // 2) - 1]
                            hist[x] -= 1
                            elems_num -= 1
                        if j + R // 2 < W:
                            x = self.img[i + t, j + R // 2]
                            hist[x] += 1
                            elems_num += 1

                self.processed[i, j] = calculate_median(hist, elems_num)


class ConstantTimeMedianFilter:
    def __init__(self, img):
        self.img = img
        self.processed = None

    def process(self, R):
        H, W = self.img.shape
        self.processed = np.zeros_like(self.img)

        hists = []
        columns = []

        R_half = R // 2

        vals_count = 0

        for i in range(W):
            hists.append(np.histogram(self.img[0 : R_half + 1, i : i + 1].reshape(1, R_half + 1)[0],
                                      bins=[x for x in range(0, 257)])[0])
            columns.append(R_half + 1)

        hist = np.zeros(pixel_vals)

        for x in range(R_half + 1):
            hist += hists[x]
            vals_count += columns[x]

        self.processed[0, 0] = calculate_median(hist, vals_count)

        for j in range(1, W):
            old = 0
            new = 0

            if j - R_half - 1 >= 0:
                old = hists[j - R_half - 1]
                vals_count -= columns[j - R_half - 1]
            if j + R_half < W:
                new = hists[j + R_half]
                vals_count += columns[j + R_half]
            hist = hist - old + new
            self.processed[0, j] = calculate_median(hist, vals_count)

        for i in range(1, H):
            hist = np.zeros(pixel_vals)
            vals_count = 0

            for r in range(R_half + 1):
                if i - R_half - 1 >= 0:
                    x = self.img[i - R_half - 1, r]
                    hists[r][x] -= 1
                    columns[r] -= 1
                if i + R_half < H:
                    x = self.img[i + R_half, r]
                    hists[r][x] += 1
                    columns[r] += 1
                hist += hists[r]
                vals_count += columns[r]

            self.processed[i, 0] = calculate_median(hist, vals_count)

            for j in range(1, W):
                new = 0
                old = 0
                if j + R_half < W:
                    if i - R_half - 1 >= 0:
                        x = self.img[i - R_half - 1, j + R_half]
                        hists[j + R_half][x] -= 1
                        columns[j + R_half] -= 1
                    if i + R_half < H:
                        x = self.img[i + R_half, j + R_half]
                        hists[j + R_half][x] += 1
                        columns[j + R_half] += 1
                    new = hists[j + R_half]
                    vals_count += columns[j + R_half]
                if j - R_half - 1 >= 0:
                    old = hists[j - R_half - 1]
                    vals_count -= columns[j - R_half - 1]
                hist = hist - old + new

                self.processed[i, j] = calculate_median(hist, vals_count)

