import numpy as np
import math
import cv2
from scipy import ndimage
from afqa_toolbox.tools import normed, visualize_orientation_field, EPS, resized


class GaborEnhancement:
    """ Class for fingerprint enhancement with oriented Gabor filers

    The functions are reimplemented in Python from the matlab implementation by Peter Kovesi
    # Peter Kovesi
    # School of Computer Science & Software Engineering
    # The University of Western Australia
    # http://www.csse.uwa.edu.au/~pk
    """
    def __init__(self):
        self.segment_blksize = 16
        self.segment_thresh = 0.1

        self.orient_gradientsigma = 1
        self.orient_blocksigma = 7
        self.orient_smoothsigma = 7

        self.ridge_freq_blksze = 38
        self.ridge_freq_windsze = 5

        self.freqest_min_wave_length = 5
        self.freqest_max_wave_length = 15

        self.filter_kx = 0.65
        self.filter_ky = 0.65
        self.filter_angle_inc = 3
        self.filter_thresh = -3

    def ridge_segment(self, img):
        img = (img - np.mean(img)) / np.std(img)

        std_img = np.zeros(shape=img.shape)
        for i in range(0, img.shape[0], self.segment_blksize):
            for j in range(0, img.shape[1], self.segment_blksize):
                std_img[i:i + self.segment_blksize, j:j + self.segment_blksize] = img[i:i + self.segment_blksize, j:j + self.segment_blksize].std()

        mask = std_img > self.segment_thresh

        img = img - img[mask].mean()
        normed_img = img / img[mask].std()
        return mask, normed_img

    def ridge_orient(self, normed_img):
        grad_x = cv2.Sobel(normed_img, cv2.CV_64F, 1, 0, ksize=5, scale=2**(2+1+0-5*2))
        grad_y = cv2.Sobel(normed_img, cv2.CV_64F, 0, 1, ksize=5, scale=2**(2+0+1-5*2))

        grad_xx = grad_x * grad_x
        grad_yy = grad_y * grad_y
        grad_xy = grad_x * grad_y

        ksize = int(round(6*self.orient_blocksigma))
        ksize = ksize+1 if ksize % 2 == 0 else ksize
        grad_xx = cv2.GaussianBlur(grad_xx, (ksize, ksize), self.orient_blocksigma)
        grad_yy = cv2.GaussianBlur(grad_yy, (ksize, ksize), self.orient_blocksigma)
        grad_xy = cv2.GaussianBlur(grad_xy, (ksize, ksize), self.orient_blocksigma)

        # Analytic solution
        denom = np.sqrt(grad_xy**2 + (grad_xx - grad_yy)**2) + EPS
        sin2theta = grad_xy/denom
        cos2theta = (grad_xx - grad_yy)/denom

        if self.orient_smoothsigma:
            ksize_smooth = int(round(6 * self.orient_smoothsigma))
            ksize_smooth = ksize_smooth + 1 if ksize_smooth % 2 == 0 else ksize_smooth

            sin2theta = cv2.GaussianBlur(sin2theta, (ksize_smooth, ksize_smooth), self.orient_smoothsigma)
            cos2theta = cv2.GaussianBlur(cos2theta, (ksize_smooth, ksize_smooth), self.orient_smoothsigma)

        orient_image = np.pi/2 + np.arctan2(sin2theta, cos2theta)/2

        moment_min = (grad_yy + grad_xx) / 2 - (grad_xx - grad_yy) * cos2theta / 2 - grad_xy * sin2theta / 2
        moment_max = (grad_yy + grad_xx) - moment_min

        reliability = 1 - moment_min / (moment_max + 0.001)
        #coherence = ((moment_max - moment_min) / (moment_max + moment_min)) ** 2

        reliability = reliability * (denom > 0.001)

        return orient_image, reliability

    def ridge_freq(self, normed_img, mask, orient_image):
        rows, cols = normed_img.shape
        freq = np.zeros((rows, cols))

        for r in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
            for c in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):
                blkim = normed_img[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]
                blkor = orient_image[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]

                freq[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze] = self.frequest(blkim, blkor)
        return freq * mask

    def frequest(self, blkim, blkor):
        rows, cols = np.shape(blkim)

        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = math.atan2(sinorient, cosorient) / 2

        # 10x Slower, but a bit prettier results.
        #rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

        center = blkim.shape[1] / 2, blkim.shape[0] / 2
        rot_mat = cv2.getRotationMatrix2D(center, orient / np.pi * 180 + 90, 1)
        rotim = cv2.warpAffine(blkim, rot_mat, (blkim.shape[1], blkim.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

        # Crop invalid regions
        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

        # Sum down the columns to get a projection of the grey values down the ridges.
        proj = np.sum(rotim, axis=0)
        dilation = ndimage.grey_dilation(proj, self.ridge_freq_windsze)#, structure=np.ones(self.ridge_freq_windsze))
        max_pts = (dilation == proj) & (proj > np.mean(proj))
        max_ind = np.where(max_pts)[0]

        if len(max_ind) >= 2:
            nb_peaks = len(max_ind)
            wave_length = (max_ind[-1] - max_ind[0]) / (nb_peaks - 1)
            if wave_length >= self.freqest_min_wave_length and wave_length <= self.freqest_max_wave_length:
                return 1 / wave_length
        return 0

    def ridge_filter(self, normed_image, orient_image, freq):
        filtered_img = np.zeros(normed_image.shape)
        rows, cols = normed_image.shape

        freq[freq > 0] = np.round(freq[freq > 0], 2)
        unfreq = np.unique(freq[freq > 0])

        freq_idx = np.ones((100,))
        for k in range(len(unfreq)):
            freq_idx[int(round(unfreq[k]*100))] = k

        filter_angle = {}
        sze = np.zeros((len(unfreq),))
        for k in range(len(unfreq)):
            sigmax = 1 / unfreq[k] * self.filter_kx
            sigmay = 1 / unfreq[k] * self.filter_ky

            sze[k] = int(np.round(3 * np.max([sigmax, sigmay])))

            x, y = np.meshgrid(np.linspace(-sze[k], sze[k], int(2*sze[k]+1)), np.linspace(-sze[k], sze[k], int(2*sze[k]+1)))

            reffilter = np.exp(-(x**2 / sigmax**2 + y**2 / sigmay**2)) * np.cos(2 * np.pi * unfreq[k] * x)

            f_center = reffilter.shape[1] / 2, reffilter.shape[0] / 2
            for o in range(180//self.filter_angle_inc):
                rot_mat = cv2.getRotationMatrix2D(f_center, -(o * self.filter_angle_inc) + 90, 1)
                rotim = cv2.warpAffine(reffilter, rot_mat, (reffilter.shape[1], reffilter.shape[0]), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                filter_angle[(k, o)] = rotim

        maxsze = int(sze[0])
        maxorientindex = np.round(180 / self.filter_angle_inc)
        orientindex = np.round((orient_image / np.pi) * (180 / self.filter_angle_inc))

        orientindex[orientindex < 1] += maxorientindex
        orientindex[orientindex > maxorientindex] -= maxorientindex

        for r in range(maxsze, rows - maxsze):
            for c in range(maxsze, cols - maxsze):
                if freq[r, c] == 0:
                    continue
                filterindex = int(freq_idx[round(freq[r, c] * 100)])

                s = int(sze[filterindex])
                img_block = normed_image[r - s:r + s + 1][:, c - s:c + s + 1]
                filtered_img[r, c] = np.sum(img_block * filter_angle[(filterindex, int(orientindex[r, c]) - 1)])

        binim = filtered_img < self.filter_thresh
        return filtered_img, binim

    def enhance(self, img, extended_data=False):
        mask, normed_image = self.ridge_segment(img)   # normalise the image and find a ROI
        orient_image, reliability = self.ridge_orient(normed_image)       # compute orientation image
        freq = self.ridge_freq(normed_image, mask, orient_image)
        filtered_image, bin_img = self.ridge_filter(normed_image, orient_image, freq)

        # Saved intermediate results
        if extended_data:
            return filtered_image, bin_img, normed_image, mask, orient_image, reliability, freq
        return filtered_image, bin_img

