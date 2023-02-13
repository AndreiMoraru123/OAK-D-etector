import torch.nn.functional as F
from utils import *

checkpoint = torch.load('checkpoints/checkpoint_ssd300.pt', map_location='cuda')
model = checkpoint['model']
model.eval()


def detect_objects(predicted_locs, predicted_scores, max_overlap, min_score, top_k):
    """
    Decipher the 8732 locations and class scores and transform them into actual detections.
    Perform Non-Maximum Suppression (NMS) on the resulting detections.
    :param predicted_locs: (N, 8732, 4)
    :param predicted_scores: (N, 8732, n_classes)
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that NMS is not applied to the smaller box
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (N, top_k, 6) -> (N, class, score, x0, y0, x1, y1)
    """

    priors_cxcy = model.priors_cxcy.to('cuda')
    n_classes = len(label_map)

    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes, all_images_labels, all_images_scores = [], [], []

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732, 4) fractional

        # Lists to store boxes and scores for this image
        image_boxes, image_labels, image_scores = [], [], []

        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where the class is the one we're looking for (above min_score)
            class_scores = predicted_scores[i][:, c]
            score_above_min_score = class_scores > min_score  # (8732)
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores, sort_ind = class_scores[score_above_min_score].sort(dim=0, descending=True)
            class_decoded_locs = decoded_locs[score_above_min_score][sort_ind]  # (n_above_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(
                class_decoded_locs, class_decoded_locs  # (n_above_min_score, n_above_min_score)
            )

            # Non-Maximum Suppression (NMS)
            # Keep only the boxes that have an IoU of less than 'max_overlap' with the previously selected boxes
            # We'll end up with only the best boxes, as the worst ones will have been removed
            supress = torch.zeros(n_above_min_score, dtype=torch.bool).to('cuda')  # (n_above_min_score)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box has already been selected for suppression, continue
                if supress[box] == 1:
                    continue

                # Suppress boxes whose IoU is greater than 'max_overlap'
                # Find such boxs and update the suppression vector
                supress = supress | (overlap[box] > max_overlap)

                # Don't suppress this box, even though it has an IoU of 'max_overlap' with itself
                supress[box] = 0

            # Store only the best (n_above_min_score - supress.sum()) boxes
            image_boxes.append(class_decoded_locs[~supress])
            image_labels.append(torch.LongTensor((~supress).sum().item() * [c]).to('cuda'))
            image_scores.append(class_scores[~supress])

        # If no object of any class was found, add a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0, 0, 1, 1]]).to('cuda'))
            image_labels.append(torch.LongTensor([0]).to('cuda'))
            image_scores.append(torch.FloatTensor([0]).to('cuda'))

        # Concatenate the best (n_above_min_score - supress.sum()) boxes for each class found in this image
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top 'k' objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)
            image_scores = image_scores[:top_k]  # (top_k)

        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores