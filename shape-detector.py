import cv2
import matplotlib.pyplot as plt 
#  This program detects shapes in a video stream from the webcam and displays the detected shapes along with their names.
#  It uses OpenCV for image processing and contour detection, and Matplotlib for displaying the results.    

def preprocess_image(frame): # Preprocess the image for shape detection
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
    blurred = cv2.GaussianBlur(gray,(5,5,), 0) # Apply Gaussian blur to the grayscale image
    _,thresh=cv2.threshold(blurred,100,255,cv2.THRESH_BINARY_INV) # Apply binary thresholding to the blurred image
    edges=cv2.Canny(blurred,50,150) # Apply Canny edge detection to the blurred image
    return gray, blurred, thresh, edges

def detect_shapes(frame,contours): # Detect shapes in the image using contours
    output=frame.copy() # Create a copy of the original frame for drawing contours
    for cnt in contours: # Iterate through each contour found in the image
        area=cv2.contourArea(cnt) # Calculate the area of the contour
        if area>300: # Filter out small contours based on area
            approx=cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True) # Approximate the contour to a polygon
            x,y,w,h=cv2.boundingRect(cnt) # Get the bounding rectangle of the contour
            vertices=len(approx) # Get the number of vertices of the approximated polygon
            if vertices==3:
                shape="Triangle"
            elif vertices==4:
                aspect_ratio=float(w)/h
                if aspect_ratio>=0.95 and aspect_ratio<=1.05:
                    shape="Square"
                else:
                    shape="Rectangle"
            elif vertices>7:
                shape="Circle"
            else:
                shape="Polygon"
            cv2.drawContours(output,[approx],0,(255,0,0),2) # Draw the contour on the output image
            cv2.putText(output,shape,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1) # Put the shape name on the output image
    return output
def display_plot(thresh,edges,output_rgb): # Display the processed images using Matplotlib
    plt.figure(figsize=(10,4)) # Create a figure for displaying images
    plt.subplot(1,3,1)
    plt.title("Thresholded Image")
    plt.imshow(thresh,cmap='gray')
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.title("Canny Edges")
    plt.imshow(edges,cmap='gray')
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.title("Detected Shapes")
    plt.imshow(output_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main(): # Main function to run the shape detection program
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened(): # Check if the webcam is opened successfully
        print("Error: Could not open webcam.")
        return
    print("Press 'q' to quit the program.")
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Error: Cannot read frame.") # If the frame cannot be read, exit the loop
            break
        frame=cv2.resize(frame,(640,480))
        gray,blurred,thresh,edges=preprocess_image(frame) # Preprocess the frame for shape detection
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Find contours in the thresholded image
        output=detect_shapes(frame,contours) # Detect shapes in the frame using the contours found
        cv2.imshow("Shape Detector", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    final_rgb=cv2.cvtColor(output,cv2.COLOR_BGR2RGB) # Convert the output image to RGB for displaying with Matplotlib
    display_plot(thresh,edges,final_rgb) # Display the processed images using Matplotlib
if __name__=="__main__":
    main()