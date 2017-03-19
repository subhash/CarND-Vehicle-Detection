from glob import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import pickle


def display_images(images, cmap=None, col=3, title=None):
    row = (len(images) - 1) // col + 1
    gs = gridspec.GridSpec(row, col)
    fig = plt.figure(figsize=(30, 10 + (row / 4) * 10))
    for im, g in zip(images, gs):
        s = fig.add_subplot(g)
        s.imshow(im, cmap=cmap)
        if title: s.set_title(title)
    gs.tight_layout(fig)
    plt.show()


class Image:
    def __init__(self, image):
        self.image = image
        self.features = self.image

    def to_color_space(self, cmap):
        cmaps = {
            'RGB': None,
            'HSV': cv2.COLOR_RGB2HSV,
            'HLS': cv2.COLOR_RGB2HLS,
            'YCrCb': cv2.COLOR_RGB2YCrCb
        }
        conv = cmaps[cmap]
        ret = Image(cv2.cvtColor(self.image, conv)) if conv else self
        return ret

    def feats(self):
        return self.features.ravel()

    #     def clip(self, from_y, to_y):
    #         return Image(self.image[from_y:to_y, :])

    def clip(self, box):
        from_x, from_y, to_x, to_y = *box.low, *box.high
        return Image(self.image[from_y:to_y, from_x:to_x])

    def copy(self):
        return Image(np.copy(self.image))

    def scale(self, ratio):
        return self.copy() if ratio == 1. else Image(cv2.resize(self.image, (0, 0), fx=ratio, fy=ratio))

    def heatmap(self, windows, thresh=2):
        h, w = self.image.shape[0], self.image.shape[1]
        heat = np.zeros((h, w))
        for w in windows:
            lx, ly, hx, hy = *w.low, *w.high
            heat[ly:hy, lx:hx] += 1
        heat[heat < thresh] = 0
        return Image(np.clip(heat, 0, 255))


class HOG(Image):
    def __init__(self, channel, pix_per_cell=8, cell_per_block=2, orient=9):
        Image.__init__(self, channel)
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.features, self.visualization = hog(channel, orientations=orient,
                                                pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                                cells_per_block=(self.cell_per_block, self.cell_per_block),
                                                transform_sqrt=True, visualise=True, feature_vector=False)

    def features_at(self, ix, iy, wx, wy):
        wx_blocks, wy_blocks = (wx // self.pix_per_cell) - 1, (wy // self.pix_per_cell) - 1
        bx, by = (ix // self.pix_per_cell), (iy // self.pix_per_cell)
        return self.features[by:by + wy_blocks, bx:bx + wx_blocks].ravel()


class SpatialBin(Image):
    def __init__(self, image, spatial_size=(32, 32)):
        Image.__init__(self, image)
        self.spatial_size = spatial_size
        self.features = cv2.resize(self.image, self.spatial_size)

    def features_at(self, ix, iy, wx, wy):
        cropped = self.image[iy:iy + wy, ix:ix + wx]
        return cv2.resize(cropped, self.spatial_size)


class ColorHistogram(Image):
    def __init__(self, image, bins=32):
        Image.__init__(self, image)
        self.bins = bins
        self.features = np.histogram(self.image, bins=self.bins)[0]

    def features_at(self, ix, iy, wx, wy):
        cropped = self.image[iy:iy + wy, ix:ix + wx]
        return np.histogram(cropped, bins=self.bins)[0]


class Box:
    def __init__(self, low, high):
        self.low = tuple(low)
        self.high = tuple(high)

    def __repr__(self):
        return "Box: %s, %s" % (self.low, self.high)

    def draw_on(self, im, color=((0, 0, 255))):
        image = im.copy()
        cv2.rectangle(image.image, self.low, self.high, color, 6)
        return image

    def clip(self, box):
        low = np.max((self.low, box.low), axis=0)
        high = np.min((self.high, box.high), axis=0)
        return Box(low, high)

    def offset(self, off):
        low = tuple(np.add(self.low, off))
        high = tuple(np.add(self.high, off))
        return Box(low, high)

    def scale(self, scale):
        diff = np.subtract(self.high, self.low)
        delta = np.subtract(np.multiply(scale, diff), diff)
        low = np.subtract(self.low, np.multiply(0.5, delta))
        high = np.add(self.high, np.multiply(0.5, delta))
        return Box(np.int_(low), np.int_(high))

    def translate(self, trans):
        low = np.add(self.low, trans)
        high = np.add(self.high, trans)
        return Box(np.int_(low), np.int_(high))

    def centroid(self):
        return np.average((self.low, self.high), axis=0)

    def distance_from(self, box):
        x1, y1 = self.centroid()
        x2, y2 = box.centroid()
        dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return dist

    def within(self, box):
        return (np.array(self.low) > box.low).all() and (np.array(self.high) < box.high).all()

    def difference(self, box):
        trans = np.subtract(self.centroid(), box.centroid())
        scale = np.subtract(self.high, self.low) / np.subtract(box.high, box.low)
        return (trans, scale)

    def area(self):
        w, h = np.subtract(self.high, self.low)
        return w * h


class Window(Box):
    def __init__(self, features, low, high):
        Box.__init__(self, low, high)
        self.features = features

    def scale(self, scale):
        lx, ly, hx, hy = *self.low, *self.high
        lx, ly, hx, hy = int(lx * scale), int(ly * scale), int(hx * scale), int(hy * scale)
        return Window(self.features, (lx, ly), (hx, hy))


class Classifier:
    def __init__(self, svc=None, scaler=None):
        self.svc = svc
        self.scaler = scaler

    def extract_feature_spaces(self, im):
        hsv = im.to_color_space('HSV').image
        ycrcb = im.to_color_space('YCrCb').image
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        y_chan, cr_chan, cb_chan = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
        spaces = [s for c in [y_chan, cr_chan, cb_chan] for s in [HOG(c), SpatialBin(c), ColorHistogram(c)]]
        return spaces

    def extract_features(self, im):
        features = [f for s in self.extract_feature_spaces(im) for f in s.feats()]
        return features

    def extract_features_at(self, spaces, x, y, wx, wy):
        features = [f for s in spaces for f in s.features_at(x, y, wx, wy).ravel()]
        return np.array(features)

    def train(self, vehicles, non_vehicles):
        vehicle_feats = [self.extract_features(v) for v in vehicles]
        non_vehicle_feats = [self.extract_features(n) for n in non_vehicles]
        X = np.vstack((vehicle_feats, non_vehicle_feats)).astype(np.float64)
        y = np.concatenate((np.ones(len(vehicle_feats)), np.zeros(len(non_vehicle_feats))))

        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=rand_state)
        self.svc = LinearSVC()
        print("Feature shape ", X_train.shape)
        self.svc.fit(X_train, y_train)
        print("Accuracy - ", self.svc.score(X_test, y_test))

    def predict(self, features):
        scaled = self.scaler.transform(features.reshape(1, -1))
        return self.svc.predict(scaled)

    def find_predictions(self, im, scales=[1.0]):
        wx, wy = 64, 64
        sx, sy = 2 * 8, 2 * 8
        windows = []
        for s in scales:
            scaled = im.scale(s)
            ix, iy = scaled.image.shape[1], scaled.image.shape[0]
            spaces = self.extract_feature_spaces(scaled)
            for x in range(0, ix - wx, sx):
                for y in range(0, iy - wy, sy):
                    feats = self.extract_features_at(spaces, x, y, wx, wy).ravel()
                    pred = self.predict(feats)
                    if pred == 1.:
                        win = Window(feats, (x, y), (x + wx, y + wy)).scale(1. / s)
                        windows.append(win)
        return windows

    def refine_predictions(self, im, windows, thresh):
        heat = im.heatmap(windows, thresh)
        labels, nlabels = label(heat.image)
        boxes = []
        for lbl in range(1, nlabels + 1):
            ly, lx = (labels == lbl).nonzero()
            box = Box((np.min(lx), np.min(ly)), (np.max(lx), np.max(ly)))
            boxes.append(box)
        return boxes

    def find_cars_debug(self, image):
        startx, starty, stopx, stopy = 0, 400, image.image.shape[1], 656
        clip_box = Box((0, starty), (stopx, stopy))
        im = image.clip(clip_box)

        windows = self.find_predictions(im, scales=[0.5, 0.66])
        boxes = self.refine_predictions(im, windows, thresh=2)
        window_image, heat_image, box_image = im.copy(), im.heatmap(windows, thresh=0), im.copy()
        for w in windows: window_image = w.draw_on(window_image)
        for b in boxes: box_image = b.draw_on(box_image)
        return [window_image, heat_image, box_image]

    #     def find_cars(self, image):
    #         startx, starty, stopx, stopy = 0, 400, image.image.shape[1], 656
    #         clip_box = Box((0, starty), (stopx, stopy))
    #         orig = Image(image)
    #         im = orig.clip(starty, stopy)

    #         windows = self.find_predictions(im, scales=[0.5, 0.66, 1.])
    #         boxes = self.refine_predictions(im, windows, thresh=2)
    #         box_image = orig.copy()
    #         for b in boxes:
    #             b = b.off((0, starty))
    #             box_image = b.draw_on(box_image)

    #         return box_image.image

    def find_boxes(self, image, find_box, scales, thresh=2):
        startx, starty = find_box.low
        im = image.clip(find_box)
        windows = self.find_predictions(im, scales=scales)
        boxes = self.refine_predictions(im, windows, thresh)
        return [b.offset((startx, starty)) for b in boxes]


class Vehicle:
    def __init__(self, id, box):
        self.id = id
        self.boxes = [box]
        self.rejected = None

    def admissible(self, box, tol):
        next_box = self.next_position()
        trans, scale = next_box.difference(box)
        tx, ty = trans
        sx, sy = scale
        trans_tol, scale_tol = tol
        trans_limit = -trans_tol < tx < trans_tol and -trans_tol < ty < trans_tol
        scale_limit = -scale_tol < sx - 1.0 < scale_tol and -scale_tol < sy - 1.0 < scale_tol
        return trans_limit and scale_limit

    def update(self, box, iteration_id, tol):
        if self.admissible(box, tol):
            self.boxes = [box] + self.boxes
            self.boxes = self.boxes[:5]
            self.rejected = None
        else:
            next_box = self.next_position()
            # self.boxes = [next_box] + self.boxes
            # self.boxes = self.boxes[:5]
            self.rejected = box

            diff = next_box.difference(box)
            print(iteration_id, ':', self.id, ': rejected - ', diff)

    def next_position(self):
        trans, scale = self.velocity()
        return self.position().scale(scale).translate(trans)

    def velocity(self):
        if (len(self.boxes) > 1):
            diff = [a.difference(b) for a, b in zip(self.boxes[:-1], self.boxes[1:])]
            ave = np.average(diff, axis=0)
            return ave
        else:
            return self.boxes[0].difference(self.boxes[0])

    def position(self):
        return self.boxes[0]

    def draw_on(self, image):
        image = self.position().draw_on(image)
        #if (self.rejected):
        #    image = self.rejected.draw_on(image, color=(255, 0, 0))
        return image


class VehicleFinder:
    def __init__(self, svc, scaler):
        self.clf = Classifier(svc, scaler)
        self.vehicles = []
        self.vehicle_id = 0
        self.iteration_id = 0
        self.false_positive = None

    def just_starting(self):
        return self.iteration_id < 3

    def create_car(self, box, im, origins):
        h, w = im.image.shape[0], im.image.shape[1]
        # corners = [Box((w-5,h-5), (w,h)), Box((0,h-5), (5,h)), Box((w//2,h//2), (w//2+5,h//2+5))]
        valid_origin = [o for o in origins if box.distance_from(o) < 200.]
        if (self.just_starting() or valid_origin):
            self.vehicle_id = self.vehicle_id + 1
            print(self.iteration_id, ': adding vehicle ', self.vehicle_id)
            self.vehicles += [Vehicle(self.vehicle_id, box)]
        else:
            self.false_positive = box
            print(self.iteration_id, ': false positive', [box.distance_from(o) for o in origins])

    def draw_cars(self, image):
        boxed_image = image.copy()
        for v in self.vehicles:
            boxed_image = v.draw_on(boxed_image)
        #if (self.false_positive):
        #    boxed_image = self.false_positive.draw_on(boxed_image, color=(0, 255, 0))
        #    self.false_positive = None
        return boxed_image

    def update_cars(self, boxes, image, origins, cent_tol=30., shape_tol=(30, 0.25)):
        cars = self.vehicles.copy()
        updated = []
        for b in boxes:
            candidates = [v for v in cars if v.position().distance_from(b) < cent_tol]
            if candidates:
                candidates[0].update(b, self.iteration_id, shape_tol)
                # cars.remove(candidates[0])
                updated.append(candidates[0])
            else:
                print(self.iteration_id, ": giving up for ", [v.position().distance_from(b) for v in cars])
                self.create_car(b, image, origins)
        for c in cars:
            if c not in updated:
                print(self.iteration_id, ': removing vehicle ', c.id)
                self.vehicles.remove(c)

    def find_existing_car(self, image, vehicle, search_box):
        find_box = vehicle.next_position().scale(2.0).clip(search_box)
        scales = [0.5, 0.75]
        return self.clf.find_boxes(image, find_box, scales, thresh=1)

    def find_cars(self, frame):
        self.iteration_id += 1
        search_box = Box((0, 400), (frame.shape[1], 656))
        image = Image(frame)
        origins = [Box((0, 400,), (230, 656)), Box((1050, 400), (1280, 656)), Box((430, 400), (850, 500))]

        if (self.just_starting()):
            scales = [0.5, 0.66, 1.0]
            boxes = self.clf.find_boxes(image, search_box, scales, thresh=2)
            self.update_cars(boxes, image, origins, cent_tol=30, shape_tol=(30, 0.25))
        else:
            vehicle_boxes = [b for v in self.vehicles for b in self.find_existing_car(image, v, search_box)]
            left_boxes = self.clf.find_boxes(image, origins[0], [0.5], thresh=1)
            right_boxes = self.clf.find_boxes(image, origins[1], [0.5], thresh=1)
            middle_boxes = []  # self.clf.find_boxes(image, origins[2], [1.5], thresh=1)
            boxes = vehicle_boxes + left_boxes + right_boxes + middle_boxes
            print(self.iteration_id, ": boxes - ", len(vehicle_boxes), len(left_boxes), len(right_boxes),
                  len(middle_boxes))
            self.update_cars(boxes, image, origins, cent_tol=100, shape_tol=(50, 0.5))

        result = self.draw_cars(image)
        #for b in origins: result = b.draw_on(result, color=(255, 255, 255))
        return result.image


def train(clf, vehicles, non_vehicles):
    vehicle_feats = [clf.extract_features(v) for v in vehicles]
    non_vehicle_feats = [clf.extract_features(n) for n in non_vehicles]
    X = np.vstack((vehicle_feats, non_vehicle_feats)).astype(np.float64)
    y = np.concatenate((np.ones(len(vehicle_feats)), np.zeros(len(non_vehicle_feats))))

    clf.scaler = StandardScaler().fit(X)
    X_scaled = clf.scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=rand_state)
    clf.svc = LinearSVC()
    print("Feature shape ", X_train.shape)
    clf.svc.fit(X_train, y_train)
    print("Accuracy - ", clf.svc.score(X_test, y_test))

    return clf.svc, clf.scaler


######### Images #####################

# vehicle_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in glob("./vehicles/**/*.png")]
# non_vehicle_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in glob("./non-vehicles/**/*.png")]
# np.random.shuffle(vehicle_images)
# np.random.shuffle(non_vehicle_images)
#
# print(len(vehicle_images), len(non_vehicle_images))

######## Training #####################

# vehicles = [Image(v) for v in vehicle_images]
# non_vehicles = [Image(v) for v in non_vehicle_images]
# print(len(vehicles), len(non_vehicles))
# t = time.time()
# svc, scaler = train(Classifier(), vehicles, non_vehicles)
# print("Time taken to train - ", time.time() - t)
#
# with open('classifier.pickle', 'wb') as handle:
#     pickle.dump(svc, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

######## Retrieve #########################

with open('classifier.pickle', 'rb') as handle:
    svc = pickle.load(handle)
    scaler = pickle.load(handle)
    print(svc, scaler)

finder = VehicleFinder(svc, scaler)


######## Video #############################

video = VideoFileClip("project_video.mp4")
marked = video.fl_image(finder.find_cars)
marked.write_videofile("project_output.mp4", audio=False)



