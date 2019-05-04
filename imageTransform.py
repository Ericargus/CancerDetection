import cv2
import random
import numpy as np 

class saltPepper():
    def __init__(self, probability = 0.01):
        self._probability = probability
        self._threshold = 1 - probability
    def __call__(self, image):
        h, w = image.shape[:2]
        noise = np.random.randn(h, w)
        image[noise < self._probability] = 0
        image[noise > self._threshold] = 1
        return image

class horizontalFlip():
    def __call__(self, image):
        factor = np.random.choice([-1, 0, 1])
        image = cv2.flip(image, factor)
        return image

class brightnessShift():
    def __init__(self, max_value = 0.1):
        self.max_value = max_value

    def __call__(self, image):
        image += np.random.uniform(- self.max_value, self.max_value)
        image = np.clip(image, 0, 1)
        return image

class brightnessScaling():
    def __init__(self, max_value = 0.08):
        self.max_value = max_value
    def __call__(self, image):
        image *= np.random.uniform(1 - self.max_value, 1 + self.max_value)
        image = np.clip(image, 0, 1)
        return image


class gammaChange():
    def __init__(self, max_value = 0.08):
        self.max_value = max_value

    def __call__(self, image):
        image = image ** (1.0/np.random.uniform(1-self.max_value, 1 + self.max_value))
        image = np.clip(image, 0, 1)
        return image

def do_elastic_transform(image, grid = 10, distort = 0.2):
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + random.uniform(-distort, distort))
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step *(1 + random.uniform(-distort, distort))
        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation = cv2.INTER_LINEAR,borderMode = cv2.BORDER_REFLECT_101,
                     borderValue = (0, 0, 0))
    return image


class ElasticDeformation():
    def __init__(self, grid = 10, max_distort = 0.15):
        self.grid = grid
        self.max_distort = max_distort

    def __call__(self, image):
        distort = np.random.uniform(0, self.max_distort)
        image = do_elastic_transform(image, self.grid, distort)
        return image


def do_rotation_transform(image, angle = 0):
    height, width = image.shape[:2]
    cc = np.cos(angle / 180 * np.pi)
    ss = np.sin(angle / 180 * np.pi)
    rotateMatrix = np.array([[cc, -ss], [ss, cc]])
    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height],], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotateMatrix.T) + np.array([width / 2, height / 2])
    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat =cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags = cv2.INTER_LINEAR,
                                 borderMode = cv2.BORDER_REFLECT_101, 
                                 borderValue = (0, 0, 0))
    return image


class rotationTransform():
    def __init__(self, angle = 15):
        self.angle = angle

    def __call__(self, image):
        angle = np.random.uniform(-self.angle, self.angle)
        image = do_rotation_transform(image, angle)
        return image


def do_horizontal_shear(image, scale = 0):
    height, width = image.shape[:2]
    dx = int(scale*width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width-dx, height], [-dx, height]], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    image = cv2.warpPerspective(image, mat, (width, height), flags = cv2.INTER_LINEAR,
                           borderMode = cv2.BORDER_REFLECT_101, borderValue = (0, 0, 0))
    return image

class horizontalShear():
    def __init__(self, max_scale = 0.2):
        self.max_scale = max_scale

    def __call__(self, image):
        scale = np.random.uniform(-self.max_scale, self.max_scale)
        image = do_horizontal_shear(image, scale = scale)
        return image

class GaussNoise():
    def __init__(self, sigma_sq = 0.1):
        self.sigma_sq = sigma_sq
    
    def __call__(self,image):
        if self.sigma_sq > 0:
            image = self._gauss_noise(image, np.random.uniform(0, self.sigma_sq))
        return image

    def _gauss_noise(self, image, sigma_sq):
        image = image.astype(np.uint32)
        h, w, c = image.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        image = image + np.stack([gauss for i in range(c)], axis = 2)
        image = np.clip(image, 0, 1)
        image = image.astype(np.float32)
        return image

                

