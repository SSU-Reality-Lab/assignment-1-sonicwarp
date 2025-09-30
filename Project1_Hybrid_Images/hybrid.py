import sys
import cv2
import numpy as np
import os

def gaussian_blur_kernel_2d(sigma, height, width):
    # 커널 초기화
    kernel = np.zeros((height,width))

    # 몫으로 중심 생성
    h_center = height / 2.0
    w_center = width / 2.0

    # 상대좌표를 구하고 각각을 가우시안 함수에 적용하여 커널 생성
    for h in range(height) :
        for w in range(width) :
            # 커널 내에서 중심과의 상대적 거리를 통해 상대좌표 구하기
            x = h - h_center
            y = w - w_center
            # 상대적 가중치만 계산
            kernel[h,w] = np.exp(-(x**2 + y**2) / (2*(sigma**2)))

    # 커널 전체를 커널 요소의 총합으로 나누어 정규화
    return kernel / np.sum(kernel)

    '''주어진 sigma와 (height x width) 차원에 해당하는 가우시안 블러 커널을
    반환합니다. width와 height는 서로 다를 수 있습니다.

    입력(Input):
        sigma:  가우시안 블러의 반경(정도)을 제어하는 파라미터.
                본 과제에서는 높이와 너비 방향으로 대칭인 원형 가우시안(등방성)을 가정합니다.
        width:  커널의 너비.
        height: 커널의 높이.

    출력(Output):
        (height x width) 크기의 커널을 반환합니다. 이 커널로 이미지를 컨볼브하면
        가우시안 블러가 적용된 결과가 나옵니다.
    '''

def cross_correlation_2d(img, kernel):
    # 커널의 높이(h), 커널의 너비(w)
    h, w = kernel.shape
    # 커널의 중심을 이미지의 모든 픽셀에 정확히 위치시키기 위하여 // 2
    h_pad = h // 2
    w_pad = w // 2
    output = np.zeros_like(img, dtype=np.float64)
    if img.ndim == 2 : # 그레이 스케일
        # 패딩은 튜플 형태, 각 차원에 얼마나 많은 패딩을 추가할 것인가 알려주는 규칙 정보
        # ((1축시작, 1축끝), (2축시작, 2축끝))
        gray_pad = ((h_pad, h_pad), (w_pad, w_pad))
        padded_img = np.pad(img, gray_pad, mode='constant')
        for i in range(img.shape[0]) : # 이미지의 높이
            for j in range(img.shape[1]) : # 이미지의 너비
                processed_gray_img = padded_img[i : i + h, j : j + w]
                output[i,j] = np.sum(processed_gray_img * kernel)
    else : # RGB 채널
        color_pad = ((h_pad, h_pad), (w_pad, w_pad), (0,0))
        padded_colored_img = np.pad(img, color_pad, mode='constant')

        for c in range(img.shape[2]) :
            for i in range(img.shape[0]) :
                for j in range(img.shape[1]) :
                    processed_colored_img = padded_colored_img[i : i + h, j : j + w, c]
                    output[i,j,c] = np.sum(processed_colored_img * kernel)

    return output

    '''주어진 커널(크기 m x n )을 사용하여 입력 이미지와의
    2D 상관(cross-correlation)을 계산합니다. 출력은 입력 이미지와 동일한 크기를
    가져야 하며, 이미지 경계 밖의 픽셀은 0이라고 가정합니다. 입력이 RGB 이미지인
    경우, 각 채널에 대해 커널을 별도로 적용해야 합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).
    '''
    
    '''출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

def convolve_2d(img, kernel):
    # 컨볼루션을 위해서 커널을 상하좌우 모두 뒤집기
    # arr[::-1] : 배열의 순서를 뒤집는 연산
    flipped_kernel = kernel[::-1,::-1]
    output = cross_correlation_2d(img, flipped_kernel)

    return output

    '''cross_correlation_2d()를 사용하여 2D 컨볼루션을 수행합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''


def low_pass(img, sigma, size):
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    filtered_img = convolve_2d(img,kernel)

    return filtered_img

    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 저역통과(low-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 저역통과 필터는 이미지의
    고주파(세밀한 디테일) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

def high_pass(img, sigma, size):
    output = img - low_pass(img,sigma,size)

    return output

    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 고역통과(high-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 고역통과 필터는 이미지의
    저주파(거친 형태) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
