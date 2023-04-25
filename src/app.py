import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import sys
import time
import torch
import torchvision

def get_ROI(frame, hand_landmarks):
    '''
    Given a frame and hand_landmarks, the function computes a bounding rectangle, uses the bounding rectangle to sample the frame, and returns the sampled ROI
    @param frame - input frame
    @param hand_landmarks - input hand landmarks
    @return ROI - ROI
    @type {(frame : np.array, hand_landmarks : mediapipe.hand_landmarks) => dict}
    '''

    rows, cols, _ = frame.shape
                        
    x_min = cols
    y_min = rows
    x_max = 0
    y_max = 0
    delta_x = 0
    delta_y = 0                   

    for n in hand_landmarks.landmark:
        x = int(n.x * cols)
        y = int(n.y * rows)

        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y

        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    x_min = np.max([x_min, 0])
    y_min = np.max([y_min, 0])
    x_max = np.min([x_max, cols])
    y_max = np.min([y_max, rows])

    delta_x = np.max([x_max - x_min, 0])
    delta_y = np.max([y_max - y_min, 0])

    multiples = 0
    for n in range(np.max(frame.shape) // 28):
        if ((n * 28) > delta_x) and ((n * 28) > delta_y):
            multiples = n
            break

    multiples += 2

    centre_x = x_min + delta_x // 2
    centre_y = y_min + delta_y // 2

    x_min = centre_x - (multiples * 28) // 2
    y_min = centre_y - (multiples * 28) // 2
    x_max = centre_x + (multiples * 28) // 2
    y_max = centre_y + (multiples * 28) // 2

    x_min = np.max([x_min, 0])
    y_min = np.max([y_min, 0])
    x_max = np.min([x_max, cols])
    y_max = np.min([y_max, rows])

    ROI = {
        'x_min' : x_min,
        'y_min' : y_min,
        'x_max' : x_max,
        'y_max' : y_max,
    }

    return ROI

def classify(frame, model, classes, device):
    '''
    Given an frame, the function classifies the ROI, and returns the predicted class
    @param frame - input frame
    @param model - neural network
    @param classes - output classes
    @param device - inference device
    @returns prediction - predicted class
    @type {(frame : np.array, model : pytorch.model, device : str) => dict}
    '''

    X = torchvision.transforms.ToTensor()(frame)
    X = X[None, :]
    X = X.to(device)
    start = time.perf_counter()
    pred = model(X)
    elapsed = (time.perf_counter() - start) * 1000
    predicted = classes[pred[0].argmax(0)]
    sm = torch.nn.Softmax(1)(pred)
    confidence = sm[0][pred[0].argmax(0)]

    result = {
        'predicted' : predicted,
        'confidence' : confidence,
        'elapsed' : elapsed
    }

    return result

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-m', '--model', default='../data/saved_models/MobileNet_V3_Small_Weights.pth', type=str, help='Model')
arg_parser.add_argument('-c', '--classes', default=['fist', 'one', 'two', 'three', 'four', 'five'], type=list, help='Classes')
arg_parser.add_argument('-d', '--device', default='cuda:0', type=str, help='Device')
arg_parser.add_argument('-dw', '--draw', default=False, type=bool, help='Draw')

args = arg_parser.parse_args()

def main():

    camera = cv.VideoCapture(0)

    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv.CAP_PROP_AUTOFOCUS, 5)
    
    for i in range(10):
        ret, frame = camera.read()

    if not ret:
        print("Error:Could not read frame.")
        return 1
       
    device = args.device

    model_path = args.model    
    model = torch.load(model_path)
    model.eval()
    model.to(device)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    classes = args.classes

    results = hands.process(frame)
    frame = cv.resize(frame, (224, 224))
    X = torchvision.transforms.ToTensor()(frame)
    X = X[None, :]
    X = X.to(device)
    _ = model(X)

    draw = args.draw
    
    key = 0
    while key != 27:

        start = time.perf_counter()
        
        ret, frame = camera.read()

        if not ret:
            continue
        
        frame = cv.flip(frame, 1)

        drawing = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                if draw:
                    mp_drawing.draw_landmarks(image=drawing, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

                ROI = get_ROI(frame, hand_landmarks)
                
                pt1 = np.array([ROI['x_min'], ROI['y_min']]).astype(np.uint32)
                pt2 = np.array([ROI['x_max'], ROI['y_max']]).astype(np.uint32)

                cv.rectangle(drawing, pt1, pt2, (0,0,255), 1)

                hand_fs = frame[ROI['y_min']:ROI['y_max'], ROI['x_min']:ROI['x_max']]
                                    
                if np.any(hand_fs) == False:
                    continue

                hand_rs = cv.resize(hand_fs, (224, 224))

                result = classify(hand_rs, model, classes, device)

                pt = np.array([ROI['x_min'], ROI['y_max']]).astype(np.uint32)
                predicted_pt = pt + np.array([0, 25]).astype(np.uint32)
                confidence_pt = predicted_pt + np.array([0, 25]).astype(np.uint32)
                elapsed_pt = confidence_pt + np.array([0, 25]).astype(np.uint32)
                
                cv.putText(drawing, f"{result['predicted']}", predicted_pt, cv.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 2)
                cv.putText(drawing, f"{result['confidence'] * 100:.2f} %", confidence_pt, cv.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 2)
                cv.putText(drawing, f"{result['elapsed']:.2f} ms", elapsed_pt, cv.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 2)

        elapsed = (time.perf_counter() - start)
        frame_rate = 1 / elapsed

        cv.putText(drawing, f"{frame_rate:.2f} fps", (25, 25), cv.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), 2)

        cv.imshow("drawing", drawing)
        key = cv.waitKey(1)

        if key == ord('s'):
            print('saved')
            cv.imwrite(f"../data/drawing_{time.time():.2f}.jpg", drawing)

    cv.destroyAllWindows()
    camera.release()

    return 0


if __name__ == "__main__":
    
    sys.exit(main())
