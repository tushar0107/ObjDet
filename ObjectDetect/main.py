import cv2
import numpy as np
import time
import imutils
import time
import os
import PySimpleGUI as sg




def main():

    sg.theme('Black')

    # define the window layout
    layout_column = [[sg.Text('Object Detection', size=(100, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='',key='image')],
              [sg.Button('Record', size=(10, 1),font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14'),
               sg.Button('Capture', size=(10, 1), font='Helvetica 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14')],]

    layout = [[sg.Column(layout_column, element_justification='center')]]
    # create the window and show it without the plot
    window = sg.Window('Object Detection',layout, location=(1000, 600),finalize=True)
    # window = sg.Window('Window Title', layout, no_titlebar=True, location=(0,0), size=(800,600), keep_on_top=True)
    window.maximize()

    # Load the YOLO model
    net = cv2.dnn.readNet('yolov3_training_last.weights','yolov3_testing.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    cap = cv2.VideoCapture(0)
    recording = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    starting_time = time.time()
    frame_id = 0
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Record':
            recording = True

        elif event == 'Stop':
            recording = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)
        
        elif event=='Capture':
            recording = True
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            frame.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            


        if recording:
            # ret, frame = cap.read()
            # imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            # window['image'].update(data=imgbytes)
            _, frame = cap.read()
            frame_id += 1
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Visualising data
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)


            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y-5), font, 1/2, color, 2)
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

        

main()