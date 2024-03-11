import os
import cv2
import numpy as np
import onnxruntime as ort

def non_maximum_suppression(boxes, scores, threshold):
    # Compute the area of the bounding boxes and sort by score
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    indices = scores.argsort()[::-1]

    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        # Find the intersection
        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h

        # Compute the ratio of the intersection over union (IoU)
        iou = intersection / (area[i] + area[indices[1:]] - intersection)

        # Keep only the bounding boxes that do not overlap significantly with the current box
        non_overlap_indices = np.where(iou <= threshold)[0]
        indices = indices[non_overlap_indices + 1]

    return keep

def draw_bounding_boxes(frame, outputs, threshold=0.01):

    detections = outputs.squeeze(0)
    boxes = []
    scores = []

    detection=detections[0]
    n= len(detections)
    print(n)

    # for i in range(n):
    #     print(detection[i])

    for detection in detections:
        cx, cy, w, h, confidence, class_score = detection

        # print("C1 : ", cx)
        # print("C2 : ", cx)
        # print("C3 : ", cx)
        # print("C4 : ", cx)
        # print("Prob : ", cx)
        # print("Class No : ", class_score)


        if confidence > threshold:
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)

    if boxes:  # Check if there are any boxes
        # Convert to numpy arrays for NMS
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Apply Non-Maximum Suppression
        indices = non_maximum_suppression(boxes, scores, 0.4)  # Adjust IoU threshold as needed

        # Draw bounding boxes after NMS
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            confidence = scores[i]
            color = (0, 255, 0)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Confidence: {confidence:.2f}"
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    # Load the ONNX model
    session = ort.InferenceSession("gesturely.onnx")

    # Start the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # Query the actual frame size
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual webcam resolution: {actual_width}x{actual_height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the expected model input size (if necessary)
        if actual_width != 640 or actual_height != 640:
            frame = cv2.resize(frame, (640, 640))

        # Convert frame to float32 and normalize
        frame = frame.astype(np.float32) / 255.0
        input_data = np.transpose(frame, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})

        # Draw bounding boxes
        frame_with_boxes = draw_bounding_boxes(frame, output[0])

        # Display the resulting frame
        cv2.imshow('Frame', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
