# Data Science for Business - Group Project 

## Data Labeling 

### How to Use the Image Annotation Tool

The following steps will guide you through using the GUI-based Image Annotation Tool for labeling images of digits.

#### Setup
1) Make sure you have **Python 3.8 or higher** installed along with the necessary libraries:
   - Run the following commands to install the dependencies:
     ```bash
     !pip install pillow
     !pip install tkinter
     ```

2) To launch the tool, simply run the provided script.

#### Instructions for Labeling
Open the labeling_image notebook. Run the code provided in the main cell. 
Once the tool is running, follow these steps to label your data (also explained in the GUI):

1) **Click "Load Images"**:  
   - A file explorer will pop up. Navigate to the folder containing your images and ensure that you **enter the folder** (not just selecting the folder itself).

2) **First Image Appears**:  
   - You will now see the first image on the screen, ready to be labeled.

3) **Draw Orientation Line**:  
   - Since the digits may have different orientations, you should first draw a line from the **center of the leftmost digit** to the **center of the rightmost digit**.
   - Click on the center of the first digit, drag the line, and release at the center of the last digit. A blue line should appear.

4) **Confirm Line**:  
   - A pop-up will ask if the line is correct. If it is, click **Yes**. If not, click **No** to redraw.

5) **Draw Rectangle Around Each Digit**:  
   - Now, for each digit, click and drag to draw a rectangle starting from the **top-left corner** of the digit (reading order). Remember:
     - The **top-left corner** should always be above and to the left of the **bottom-right corner**, regardless of the digit's orientation.

6) **Label Each Digit**:  
   - After drawing each rectangle, a pop-up will ask you to enter the digit's label (from 0 to 9). Type the number and press **Enter** or click **yes**.

7) **Check Label**:  
   - You should see the labeled information appear in the CSV content area. If you zoom in or out, you’ll see the number appear on the image.

8) **Repeat for All Digits**:  
   - Label all 11 digits in the image. When finished, click **Next Image** to move to the next one.

#### Output
Once you're done, the tool will generate:
- A **CSV file** named `annotations.csv` containing all labeled data, formatted as follows:
   - **"Image Name"**, **"Position"**, **"Label"**, **"Top-Left X"**, **"Top-Left Y"**, **"Bottom-Right X"**, **"Bottom-Right Y"**  
     where:
     - *Image Name* corresponds to the original name of the image.
     - *Position* is the sequential position of the digit in the image.
     - *Label* is the digit value (0-9).
     - *Top-Left X, Y* are the coordinates of the top-left corner of the rectangle.
     - *Bottom-Right X, Y* are the coordinates of the bottom-right corner.
   
- A **"result" folder** containing all annotated images with the rectangles and labels drawn.

#### Navigation Controls
- **Zoom In**: Use the **`+`** key.
- **Zoom Out**: Use the **`-`** key.
- **Move Image**: Use the **arrow keys** to pan in any direction.

#### Quick Summary
1) **Load Images** → 2) **Draw Orientation Line** → 3) **Draw Rectangle for Each Digit** → 4) **Enter Digit Label** → 5) **Move to Next Image**

Follow this flow, and you'll quickly annotate all your images!


### How to Check the Labeling of your Image 

Open the check_labeled_image notebook. Choose the image you want to check and change the name of the image in the corresponding cell. When you changed the name of the image, run the code in the main cell to plot the original image and the labeled image. You can now compare the result of the labeling. 

_Note: the code in this notebook assume that you have a folder called 'result' with the labeled images created using the instruction above and a folder called 'images' with the original images (before labeling)._

