import re
import math

import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torchvision.transforms import v2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from Levenshtein import distance


# detection
INPUT_RESOLUTION = (800, 800)
MIN_CONF_SCORE = 0.0

# postprocessing
DBSCAN_PARAMS = {
    'eps': 6.5e-3,
    'min_samples': 1,
    'metric': 'l1',
}
MERGE_PARAMS = {
    'x_thresh': 0.005,
}

# comparison params
SIMILARITY_THRESH = 0.7


class Detector:
    """
    Faster R-CNN model trained to detect words
    """
    def __init__(self, path: str, device: str = 'cpu',
                 model_resolution: tuple[int, int] = INPUT_RESOLUTION, min_conf_score: float = MIN_CONF_SCORE):
        self.model = torch.load(path, weights_only=False, map_location=device).to(device).eval()
        self.model_resolution = model_resolution
        self.min_conf_score = min_conf_score


    def predict(self, image: np.ndarray) -> np.ndarray:
        """
            Detect bounding boxes
        Args:
            image: RGB image with shape (resolution[0], resolution[1], 3)

        Returns:
            bounding boxes coordinates corresponding to original (input) resolution
        """
        # calculate scaling vector
        IMG_HEIGHT, IMG_WIDTH, _ = image.shape
        BBOX_IMG_H, BBOX_IMG_W = self.model_resolution
        x_scale = IMG_WIDTH / BBOX_IMG_W
        y_scale = IMG_HEIGHT / BBOX_IMG_H
        scale_vector = [x_scale, y_scale, x_scale, y_scale]

        # preprocessing
        preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.model_resolution),
        ])
        input = preprocess(image)
        input = input.unsqueeze(0)

        # prediction
        prediction = self.model(input)
        boxes = prediction[0]['boxes'].detach().to('cpu').numpy()
        scores = prediction[0]['scores'].detach().to('cpu').numpy()
        boxes = boxes[scores > self.min_conf_score]

        boxes = boxes * scale_vector

        return boxes


class Recogniser:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device=device).eval()
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.device = device

    def predict(self, image: np.ndarray, bboxes: np.ndarray) -> list[str]:
        """
        Processes given image in specified areas using TrOCR
        Args:
            image: RGB image of shape (H, W, 3) with any resolution
            bboxes: numpy array of shape (n_samples, 4)

        Returns:
            list of recognised texts
        """
        # crop ROIs from original image based on scaled bboxes
        images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(el) for el in bbox]
            images.append(image[y1:y2, x1:x2, :])
        pix_val = self.processor(images=images, return_tensors='pt').pixel_values.to(device=self.device)
        gen_ids = self.model.generate(pix_val, num_beams=1)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts


class Pipeline:
    def __init__(self, detector: Detector, recogniser: Recogniser):
        self.detector = detector
        self.recogniser = recogniser

    def predict(self, images: list[np.ndarray], acceptable_names: list[list[str]]) -> dict:
        """
        Finds present students based on images and acceptable names.
        Distributes given names across all images. Empty name parts are ignored.
        Args:
            images: list of RGB images of the shape (H, W, 3) with any resolution
            acceptable_names: list of acceptable names (each consists of several name parts)

        Returns:
            dictionary with 'ids', 'scores' and 'meta'
        """
        # processing
        all_texts = []
        all_clusters = []
        for image in images:
            bboxes = self.detector.predict(image)
            img_clusters = self._find_clusters(bboxes, image.shape[:2])
            img_clusters = [self._merge_bboxes(cluster, image.shape[:2]) for cluster in img_clusters]
            all_clusters.append(img_clusters)

            for cluster in img_clusters:
                cluster_texts = self.recogniser.predict(image, cluster)
                all_texts.append(cluster_texts)

        def process_string(s: str) -> str:
            return re.sub(r'[^а-яА-Яa-zA-Z]', '', s.lower())

        # preprocessing
        processed_acceptable_names = [[process_string(name_part) for name_part in name] for name in acceptable_names]
        all_texts = [[process_string(token) for elem in cluster for token in elem.split()] for cluster in all_texts]

        sim_matrix = self._calculate_similarity_matrix(processed_acceptable_names, all_texts)

        # get recognised ids
        cluster_dict = {}
        while sim_matrix.shape[1] and sim_matrix.shape[0] and np.max(sim_matrix):
            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            cluster_dict[j.item()] = {
                'name_id': i.item(),
                'score': sim_matrix[i, j].item(),
            }
            sim_matrix[:, j] = np.zeros(sim_matrix.shape[0], dtype=np.float32)
            sim_matrix[i, :] = np.zeros(sim_matrix.shape[1], dtype=np.float32)

        result = {
            'ids': tuple(cluster_dict[cluster_id]['name_id'] for cluster_id in cluster_dict.keys()),
            'scores': tuple(cluster_dict[cluster_id]['score'] for cluster_id in cluster_dict.keys()),
        }

        result_meta = []
        n = 0
        for i in range(len(images)):
            matched_clusters = []
            matched_names = []
            unmatched_clusters = []
            scores = []
            for j, cluster in enumerate(all_clusters[i], n):
                if j in cluster_dict.keys():
                    matched_clusters.append(cluster)
                    matched_names.append(tuple(acceptable_names[cluster_dict[j]['name_id']]))
                    scores.append(cluster_dict[j]['score'])
                else:
                    unmatched_clusters.append(cluster)

            result_meta.append(
                {
                    'image_number_in_group': i,
                    'matched': {
                        'clusters': tuple([tuple([tuple([coord.item() for coord in bbox]) for bbox in cluster])
                                           for cluster in matched_clusters]),
                        'scores': tuple(scores),
                        'names': tuple(matched_names),
                    },
                    'unmatched': {
                        'clusters': tuple([tuple([tuple([coord.item() for coord in bbox]) for bbox in cluster])
                                           for cluster in unmatched_clusters]),
                    }
                }
            )
            n += len(all_clusters[i])
        result['meta'] = tuple(result_meta)


        return result

    def _merge_bboxes(self, bboxes: np.ndarray, resolution: tuple[int, ...]) -> np.ndarray:
        """
        Merge boxes in a given list based on their position
        Args:
            bboxes: list of bboxes of shape (n_samples, 4) in XYXY format
            resolution: (HEIGHT, WIDTH)

        Returns:
            merged bboxes (n_after_merge, 4)
        """

        def are_neighbours(a: np.ndarray, b: np.ndarray, thresh: int) -> bool:
            """Check whether a and b should be merged during postprocessing stage"""
            if a[0] < b[0]:
                left = a
                right = b
            else:
                left = b
                right = a

            if right[0] - left[2] < thresh:
                return True

            return False

        def get_merged_box(a):
            """Returns merged bbox based on a given list of bboxes"""
            x1 = min([bbox[0] for bbox in a])
            x2 = max([bbox[2] for bbox in a])
            y1 = sum([bbox[1] for bbox in a]) / len(a)
            y2 = sum([bbox[3] for bbox in a]) / len(a)

            return np.array([x1, y1, x2, y2], dtype=np.uint32)

        x_thresh = math.ceil(MERGE_PARAMS['x_thresh'] * resolution[1])

        unprocessed = [bboxes[i] for i in range(len(bboxes))]
        processed = []
        while len(unprocessed) > 1:
            neighbours = [unprocessed[0]]
            for i in range(len(unprocessed) - 1, 0, -1):
                if are_neighbours(unprocessed[0], unprocessed[i], x_thresh):
                    neighbours.append(unprocessed[i])
                    del unprocessed[i]

            if len(neighbours) > 1:
                merged = get_merged_box(neighbours)
                unprocessed.append(merged)
            else:
                processed.append(unprocessed[0])
            del unprocessed[0]

        processed.append(unprocessed[0])
        res = np.vstack(processed)

        return res


    def _find_clusters(self, bboxes: np.ndarray, resolution: tuple[int, ...]) -> list[np.ndarray]:
        """
        Finds groups of bboxes representing the same student name based on avg_y of bbox
        Args:
            bboxes: numpy array of shape (n_bboxes, 4) in XYXY format
            resolution: (HEIGHT, WIDTH)

        Returns:
            clusters
        """
        # update epsilon based on resolution
        specific_params = DBSCAN_PARAMS.copy()
        specific_params['eps'] *= resolution[0]

        clustering = DBSCAN(**specific_params)
        y1 = bboxes[:, 1]
        y2 = bboxes[:, 3]
        pos = (y1 + y2) / 2

        clustering.fit(pos.reshape(-1, 1))
        boxes_clusters = []
        for label in np.unique(clustering.labels_):
            boxes_i = bboxes[clustering.labels_ == label, :]

            # sorting bboxes based on x1 value
            indices = boxes_i[:, 0].argsort()
            sorted_bboxes = boxes_i[indices, :]
            boxes_clusters.append(sorted_bboxes)

        return boxes_clusters


    def _calculate_similarity_matrix(self, acceptable_tg: list[list[str]], recognised_tg: list[list[str]]) -> np.ndarray:
        """
        Calculates similarity matrix between acceptable and recognised token groups. Case-sensitive.
        Args:
            acceptable_tg: list of acceptable token groups (each consists of several tokens)
            recognised_tg: list of recognised token groups (each consists of several tokens)
        Returns:
            matrix of shape (N_acceptable_tokens, N_recognised_tokens)
        """
        mat_outer = np.zeros((len(acceptable_tg), len(recognised_tg)), dtype=np.float32)
        for i, name in enumerate(acceptable_tg):
            for j, cluster in enumerate(recognised_tg):
                mat_inner = np.zeros((len(name), len(cluster)), dtype=np.float32)
                at_least_one = False
                for m, name_part in enumerate(name):
                    for n, token in enumerate(cluster):
                        if len(name_part):
                            d = distance(token, name_part) / len(name_part)
                        else:
                            d = float('inf')
                        if 1 - d > SIMILARITY_THRESH:
                            mat_inner[m, n] = 1 - d
                            at_least_one = True
                if at_least_one:
                    while mat_inner.shape[1] and mat_inner.shape[0] and np.max(mat_inner):
                        i_0, i_1 = np.unravel_index(np.argmax(mat_inner), mat_inner.shape)
                        mat_outer[i, j] += mat_inner[i_0, i_1]
                        mat_inner = np.delete(mat_inner, i_0, 0)
                        mat_inner = np.delete(mat_inner, i_1, 1)
                else:
                    continue

        return mat_outer
