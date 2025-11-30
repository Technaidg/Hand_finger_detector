import cv2
import numpy as np
import threading
import mediapipe as mp
import time
import math
import serial
import serial.tools.list_ports

#References 
#https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
#https://medium.com/@iamramzan/finger-counter-using-opencv-and-mediapipe-a142e7faeae4


class DrawImage:
    def __init__(self):
        global ser
        # Hold the hand's data so all its details are in one place.
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.xpixelsize = 1920
        self.ypixelsize = 1080

        self.frames_elapsed = 0
        self.lefthand = np.zeros((21,3))
        self.righthand = np.zeros((21,3))

        self.threadcooldown = 0
        self.threadmaxcooldown = 100
        
    def run(self):
        global ser

        screen = cv2.VideoCapture(0)
        screen.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Try to reduce latency
        
        if not screen.isOpened():
            print("couldn't open camera")
            exit()
        #cv2.namedWindow("Hikvision Stream", cv2.WND_PROP_FULLSCREEN)
        #cv2.resizeWindow("Hikvision Stream", 1920, 1080)
        self.xpixelsize  = int(screen.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ypixelsize = int(screen.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with self.mp_hands.Hands(
        static_image_mode=False,       # For video stream
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
            pTime = 0
            cTime = 0   
            cooldown = 0
            maxcooldown = 100
            while True:
                ret, frame = screen.read()
                if not ret:
                    print("Failed to grab frame")
                    continue    
                if(cooldown > 0):
                    cooldown -=1

                small_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                #threadimghands = threading.Thread(target=self.drawhands, args=(frame, results))
                #threadimghands.start()
                self.drawhands(frame, results, small_frame)


                #cTime = time.time()
                #fps = 1/(cTime - pTime)
                #pTime = cTime

                #cv2.putText(frame, "FPS : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 8), 2)
                #self.frames_elapsed += 1


                #self.drawhands(frame, results)
                self.showtargetimage(cv2, frame)

                # Press 'c' to do smt
                key = cv2.waitKey(1) & 0xFF
                if key == -1:
                    continue  

                elif key == ord('k') and cooldown == 0:
                    print("pressed k")
                    cooldown = maxcooldown
                    
                # Press 'q' to quit
                elif key == ord('q'):
                    break

                else:
                    continue
                
                
                
            
        screen.release()
        cv2.destroyAllWindows()

    def showtargetimage(self, image, frame):
        image.imshow("Hikvision Stream", frame)
        
    def drawhands(self, frame, results, small_frame):
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            # Rescale landmarks to original frame
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                for lm in hand_landmarks.landmark:
                    lm.x *= frame.shape[1] / self.xpixelsize
                    lm.y *= frame.shape[0] / self.ypixelsize
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                fin = ''

                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                confidence_score = handedness.classification[0].score
                if confidence_score < 0.75:
                    continue
                self.updatehand(hand_landmarks, hand_label, small_frame)
                #hand indexes values
                HAND_LANDMARKS = {
                    "WRIST": 0,

                    "THUMB_CMC": 1,
                    "THUMB_MCP": 2,
                    "THUMB_IP": 3,
                    "THUMB_TIP": 4,

                    "INDEX_FINGER_MCP": 5,
                    "INDEX_FINGER_PIP": 6,
                    "INDEX_FINGER_DIP": 7,
                    "INDEX_FINGER_TIP": 8,

                    "MIDDLE_FINGER_MCP": 9,
                    "MIDDLE_FINGER_PIP": 10,
                    "MIDDLE_FINGER_DIP": 11,
                    "MIDDLE_FINGER_TIP": 12,

                    "RING_FINGER_MCP": 13,
                    "RING_FINGER_PIP": 14,
                    "RING_FINGER_DIP": 15,
                    "RING_FINGER_TIP": 16,

                    "PINKY_MCP": 17,
                    "PINKY_PIP": 18,
                    "PINKY_DIP": 19,
                    "PINKY_TIP": 20,
                }
                
                if self.lefthand[HAND_LANDMARKS["THUMB_TIP"]][0] < self.lefthand[HAND_LANDMARKS["THUMB_IP"]][0] and \
                self.lefthand[HAND_LANDMARKS["THUMB_TIP"]][1] > self.lefthand[HAND_LANDMARKS["THUMB_CMC"]][1] or \
                self.lefthand[HAND_LANDMARKS["THUMB_MCP"]][0] > self.lefthand[HAND_LANDMARKS["THUMB_TIP"]][0]:
                    val1 = 0  # Thumb finger is not raised
                else:
                    val1 = 1  # Thumb finger is raised
                    fin += 'Thumb '  # Add "Thumb" to the raised fingers string


                if self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["INDEX_FINGER_DIP"]][1] and \
                self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1] and \
                self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1]:
                    val2 = 0  # Index finger is not raised
                else:
                    val2 = 1  # Index finger is raised
                    fin += 'Index '  # Add "Index" to the raised fingers string

                if self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_DIP"]][1] and \
                self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_MCP"]][1]:
                    val3 = 0 
                else:
                    val3 = 1 
                    fin += 'Middle '  
                
                if self.lefthand[HAND_LANDMARKS["RING_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["RING_FINGER_DIP"]][1] and \
                self.lefthand[HAND_LANDMARKS["RING_FINGER_TIP"]][1] > self.lefthand[HAND_LANDMARKS["RING_FINGER_MCP"]][1]:
                    val4 = 0 
                else:
                    val4 = 1 
                    fin += 'Ring '  
                
                if self.lefthand[HAND_LANDMARKS["PINKY_TIP"]][1] > self.lefthand[HAND_LANDMARKS["PINKY_DIP"]][1] and \
                self.lefthand[HAND_LANDMARKS["PINKY_TIP"]][1] > self.lefthand[HAND_LANDMARKS["PINKY_MCP"]][1]:
                    val5 = 0 
                else:
                    val5 = 1 
                    fin += 'Pinky ' 
                
                shape = ''

                betweenleftindexandthumb = self.CalculateAngle(self.lefthand[HAND_LANDMARKS["THUMB_MCP"]][0],self.lefthand[HAND_LANDMARKS["THUMB_MCP"]][1], \
                            self.lefthand[HAND_LANDMARKS["THUMB_TIP"]][0],self.lefthand[HAND_LANDMARKS["THUMB_TIP"]][1], \
                            self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][0], self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1])
                if(betweenleftindexandthumb >= 60 and betweenleftindexandthumb <=130 and val2 == 1 and val1 == 1):
                    shape += 'LShape '  

                betweenleftindexandmiddle =   self.CalculateAngle(self.lefthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][0],self.lefthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1], \
                            self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][0],self.lefthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1], \
                            self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][0], self.lefthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1])

                if(betweenleftindexandmiddle <= 50 and betweenleftindexandmiddle > 10 and val2 == 1 and val3 == 1):
                    shape += 'VShape'

                if(val1==0 and val2==0 and val3==1 and val4==0 and val5==0):
                    shape+= 'Fuck You'

                # Calculate the total number of raised fingers
                val = val2 + val3 + val4 + val5 + val1

                # Display the total number of raised fingers on the image
                fps = str(val) + ' Lfingers'

                data = f'Fingers: {fps}, Shapes: {shape} \n'
                if self.threadcooldown <= 0:
                    thread2 = threading.Thread(target=sendData,args=(data,), daemon=True)
                    thread2.start()
                    self.threadcooldown = self.threadmaxcooldown
                else:
                    self.threadcooldown -= 1
                
                cv2.putText(frame, fps, (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

                cv2.putText(frame, (f' {confidence_score * 100:.2f}%'), (0, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                
                # Display the names of the raised fingers on the image
                cv2.putText(frame, fin, (0, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 0), 2)

                cv2.putText(frame, shape, (0, 350), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
            
            #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            #Right hand
                if self.righthand[HAND_LANDMARKS["THUMB_TIP"]][0] > self.righthand[HAND_LANDMARKS["THUMB_IP"]][0] and \
                self.righthand[HAND_LANDMARKS["THUMB_TIP"]][1] > self.righthand[HAND_LANDMARKS["THUMB_CMC"]][1] or \
                self.righthand[HAND_LANDMARKS["THUMB_MCP"]][0] < self.righthand[HAND_LANDMARKS["THUMB_TIP"]][0]:
                    val1 = 0  # Thumb finger is not raised
                else:
                    val1 = 1  # Thumb finger is raised
                    fin += 'Thumb '  # Add "Thumb" to the raised fingers string


                if self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["INDEX_FINGER_DIP"]][1] and \
                self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1] and \
                self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1]:
                    val2 = 0  # Index finger is not raised
                else:
                    val2 = 1  # Index finger is raised
                    fin += 'Index '  # Add "Index" to the raised fingers string

                if self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_DIP"]][1] and \
                self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_MCP"]][1]:
                    val3 = 0 
                else:
                    val3 = 1 
                    fin += 'Middle '  
                
                if self.righthand[HAND_LANDMARKS["RING_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["RING_FINGER_DIP"]][1] and \
                self.righthand[HAND_LANDMARKS["RING_FINGER_TIP"]][1] > self.righthand[HAND_LANDMARKS["RING_FINGER_MCP"]][1]:
                    val4 = 0 
                else:
                    val4 = 1 
                    fin += 'Ring '  
                
                if self.righthand[HAND_LANDMARKS["PINKY_TIP"]][1] > self.righthand[HAND_LANDMARKS["PINKY_DIP"]][1] and \
                self.righthand[HAND_LANDMARKS["PINKY_TIP"]][1] > self.righthand[HAND_LANDMARKS["PINKY_MCP"]][1]:
                    val5 = 0 
                else:
                    val5 = 1 
                    fin += 'Pinky ' 
                
                shape = ''

                betweenrightindexandthumb = self.CalculateAngle(self.righthand[HAND_LANDMARKS["THUMB_MCP"]][0],self.righthand[HAND_LANDMARKS["THUMB_MCP"]][1], \
                            self.righthand[HAND_LANDMARKS["THUMB_TIP"]][0],self.righthand[HAND_LANDMARKS["THUMB_TIP"]][1], \
                            self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][0], self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1])
                if(betweenrightindexandthumb >= 60 and betweenrightindexandthumb <=130 and val2 == 1 and val1 == 1):
                    shape += 'LShape '  

                betweenrightindexandmiddle =   self.CalculateAngle(self.righthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][0],self.righthand[HAND_LANDMARKS["INDEX_FINGER_MCP"]][1], \
                            self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][0],self.righthand[HAND_LANDMARKS["INDEX_FINGER_TIP"]][1], \
                            self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][0], self.righthand[HAND_LANDMARKS["MIDDLE_FINGER_TIP"]][1])
                if(betweenrightindexandmiddle <= 170 and betweenrightindexandmiddle > 140 and val2 == 1 and val3 == 1):
                    shape += 'VShape '

                if(val1==0 and val2==0 and val3==1 and val4==0 and val5==0):
                   shape+= 'Fuck You'

                # Calculate the total number of raised fingers
                val = val2 + val3 + val4 + val5 + val1

                # Display the total number of raised fingers on the image
                fps = str(val) + ' Rfingers'

                data = f'Fingers: {fps}, Shapes: {shape} \n'
                if self.threadcooldown <= 0:
                    thread2 = threading.Thread(target=sendData,args=(data,), daemon=True)
                    thread2.start()
                    self.threadcooldown = self.threadmaxcooldown
                else:
                    self.threadcooldown -= 1
                cv2.putText(frame, fps, (0, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

                cv2.putText(frame,  (f' {confidence_score * 100:.2f}%'), (0, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
                # Display the names of the raised fingers on the image
                cv2.putText(frame, fin, (0, 500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 0), 2)

                cv2.putText(frame, shape, (0, 550), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
    
    def CalculateAngle(self, P1X, P1Y, P2X, P2Y, P3X, P3Y):
        #will calculate the angle at the point P1
        #P1X = P1X * 100
        #P1Y = P1Y * 100
        #P2X = P2X * 100
        #P2Y = P2Y * 100
        #P3X = P3X * 100
        #P3Y = P3Y * 100
        
        #print(f"P1X {P1X}")
        #print(f"P1Y {P1Y}")


        numerator = P2Y*(P1X-P3X) + P1Y*(P3X-P2X) + P3Y*(P2X-P1X)
        denominator = (P2X-P1X)*(P1X-P3X) + (P2Y-P1Y)*(P1Y-P3Y)
        ratio = numerator/denominator
        angleRad = math.atan(ratio)
        angleDeg = (angleRad*180)/math.pi

        if(angleDeg<0):
            angleDeg += 180
        return angleDeg
    
    def updatehand(self, hand_landmarks, which_hand, small_frame):
        h, w, _ = small_frame.shape
        if(which_hand == 'Left'):
            for i in range(21):
                l = hand_landmarks.landmark[i]
                self.lefthand[i] = [int(l.x * w), int(l.y * h), int(l.z*1000)]
                #print(f"{which_hand} {i}: {self.lefthand[i]}")
        elif(which_hand == 'Right'):
            for i in range(21):
                l = hand_landmarks.landmark[i]
                self.righthand[i] = [int(l.x * w), int(l.y * h), int(l.z*1000)]
                #print(f"{which_hand} {i}: {self.lefthand[i]}")
                
di = DrawImage()
portopen = False
ser = None
last_port = None

def find_serial_port():
    global portopen
    global ser
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        portopen = False
        ser = None
        raise Exception("No serial ports found.")

    if ser is None:
        print("Available ports:")
        for p in ports:
            print(f"  {p.device}")

    # Use the first available port
    return ports[0].device

def connect_port():
    global portopen
    global ser
    global last_port

    while True:
        try:
            port_name = find_serial_port()

            if not portopen or ser is None:
                print(f"Connecting to {port_name}...")
                ser = serial.Serial(
                    port=port_name,
                    baudrate=9600,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.1
                )
                portopen = True
                last_port = port_name
        except Exception as e:
            portopen = False
            ser = None
            print(f"Error: {e}")
        time.sleep(5)

def sendData(data):
    global portopen
    try:
        if portopen and ser:
            ser.write(data.encode())
            ser.flush()
            print("Data sent")
        else:
            print("serial not ready")
    except Exception as e:
        print(f"Error: Couldn't send serial data: {e}")
        portopen=False
        

threadconnect = threading.Thread(target=connect_port, daemon=True).start()

thread1 = threading.Thread(target=di.run)

thread1.start()
thread1.join()

if portopen:
    ser.close()
    print("Serial closed")