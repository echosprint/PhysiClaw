# Physiclaw VLM-Based Tap System — Implementation Plan

## What This System Does

A robotic arm with a capacitive stylus taps UI elements on a phone screen. A top-down camera sees the phone. A VLM (Claude with vision) decides what to tap and where. The system requires no ADB, no app installed on the phone, no phone screenshot capture. Everything works through one camera looking down at the phone on a desk.

## The Core Idea

The VLM looks at the camera photo and reports where a target is as a percentage of the phone screen (e.g., "38% from left, 82% from top"). A one-time grid calibration maps those percentages to physical arm coordinates. A self-correction loop draws the predicted bounding box on the image and shows it back to the VLM to verify before tapping.

---

## Phase 1: One-Time Calibration

### Step 1.1 — Display Calibration Dots on Phone

Open a fullscreen webpage on the phone that shows a white background with red dots at evenly spaced interior grid positions — like a # pattern, with no dots on the edges. The dots appear at 25%, 50%, 75% horizontally and 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5% vertically — 21 dots total (3 columns × 7 rows). This keeps dots away from the phone bezel where camera distortion is worst. Each dot should be a solid red circle, large enough to be clearly visible from the camera but small enough not to overlap its neighbors. Use fullscreen mode so there is no browser toolbar.

### Step 1.2 — Take a Photo of the Dots

Use the top-down camera to photograph the phone while it displays the dots. Save this photo. The photo will show the phone with red dots on its screen, surrounded by desk background.

### Step 1.3 — Detect Dot Positions with OpenCV

The goal is to find the pixel coordinates of every red dot in the camera photo. Since we control the design (red dots on white background), detection is simple.

**Isolate the red pixels.** Convert the camera photo to HSV color space. Threshold for red to create a binary mask. Red wraps around in HSV, so two hue ranges are needed: 0-10 and 170-180, both with reasonable saturation and value minimums. The result is a binary mask where white pixels are red dots and black pixels are everything else (white background, desk, phone frame).

**Find dot centers.** Find contours or connected components of the white blobs in the mask. Compute the centroid of each one. Filter out any blobs that are too large or too small to be a dot — this removes noise and any red objects on the desk. The remaining centroids are the dot positions in camera pixel coordinates.

**Sort into grid order.** Sort all detected dots by y-coordinate first to group them into rows, then by x-coordinate within each row to order them into columns. Now dot (0,0) is the top-left dot, dot (1,0) is one column to the right, and so on. Each dot's grid position directly gives its screen percentage. For example, the dot at column 2 row 4 (counting from top-left, starting at 1) corresponds to 50% from left and 50% from top of the phone screen.

### Step 1.4 — Move Arm to Each Dot Position

Move the robotic arm so the stylus tip is directly above each detected dot. Record the GRBL machine coordinates (x mm, y mm) at each position. This can be done manually for a few key points (corners and edges) and the rest can be interpolated, or automated if the arm can be guided to each point.

### Step 1.5 — Store the Calibration

Save all the mappings to a JSON file. Each dot should have three values stored together: its screen percentage, its pixel position in the camera image, and its GRBL machine coordinates. Also store the four screen corner positions in camera pixel space.

### Step 1.6 — Build Coordinate Transform

From the stored calibration points, compute two affine transforms. The first converts screen percentages to camera pixel positions (used for drawing verification boxes). The second converts screen percentages to GRBL millimeter positions (used for moving the arm). Use OpenCV's estimateAffine2D or similar to fit these transforms from the calibration point pairs.

---

## Phase 2: Runtime — Tap Any Target

### Step 2.1 — Capture a Photo

Take a photo from the top-down camera. The photo shows the phone on the desk with whatever is currently on the screen. Do not crop or process the image. Send the raw photo.

### Step 2.2 — Ask the VLM Where the Target Is

Send the raw camera photo to the VLM (Claude with vision) along with a prompt. The prompt should tell the VLM:

- You are looking at a top-down photo of a phone on a desk.
- The phone screen is visible in the image.
- I need to tap a specific target (name or describe the target).
- Report the target's bounding box as percentages of the phone screen only.
- 0% means the left or top edge of the phone screen. 100% means the right or bottom edge.
- Return left percentage, right percentage, top percentage, and bottom percentage.

The VLM should return something like: left 38%, right 48%, top 82%, bottom 88%. These percentages describe where the target is within the phone screen, ignoring the desk and everything else in the photo.

### Step 2.3 — Draw the Bounding Box on the Photo

Using the calibration data, convert the VLM's screen percentages to camera pixel coordinates. Draw a green rectangle at that position on the original camera photo. This rectangle should visually overlap where the target is on the phone screen in the photo.

### Step 2.4 — Show the Marked Photo Back to the VLM

Send the photo with the green rectangle drawn on it back to the VLM. Ask:

- Does the green rectangle correctly cover the target on the phone screen?
- If yes, confirm it is correct.
- If no, provide corrected percentages.

This is the self-correction step. The VLM is better at judging "is this box in the right place" (verification) than "where exactly is this element" (generation). By drawing the box and showing it back, accuracy improves significantly.

### Step 2.5 — Repeat if Needed

If the VLM says the box is wrong and provides corrected percentages, draw a new box with the corrected percentages and show it back again. Repeat up to 3 rounds maximum. If after 3 rounds the VLM still says incorrect, proceed with the best estimate.

### Step 2.6 — Move the Arm and Tap

Once the VLM confirms the bounding box is correct, compute the center of the bounding box in screen percentages. Convert that center percentage to GRBL machine coordinates using the calibration transform. Move the arm to that position, lower the stylus to touch the screen, then raise the stylus.

### Step 2.7 — Verify the Tap Result

Take a new photo after tapping. Send it to the VLM and ask: "I just tapped the target. Did the screen change in a way that indicates the tap was successful?" For example, a new page opened, a menu appeared, or a button changed state. If successful, move on to the next action. If not, go back to step 2.1 and try again.

---

## What Each Component Does

**Top-down camera** — Captures photos of the phone on the desk. Used during calibration and at every step of the runtime loop.

**OpenCV** — Used only during calibration to detect red dot positions. Not used at runtime.

**Calibration JSON** — Stores the mapping between screen percentages, camera pixels, and GRBL coordinates. Created once, used forever (unless the phone or camera moves).

**Coordinate mapper** — Two affine transforms computed from calibration data. One converts screen percentages to camera pixels (for drawing boxes). One converts screen percentages to GRBL mm (for moving the arm).

**VLM (Claude with vision)** — The brain of the system. Does three jobs at runtime: (1) identify where a target is on the phone screen as percentages, (2) verify whether a drawn bounding box is correct, (3) judge whether a tap succeeded.

**GRBL arm with stylus** — The physical actuator. Receives (x, y) coordinates in mm, moves there, and taps.

---

## What This System Does NOT Need

- YOLOX or any object detection model
- ADB or USB connection to the phone
- Phone screenshot capture
- Any app installed on the phone
- Screen cropping or perspective correction at runtime
- Stylus tip detection at runtime

---

## Why This Design Works

The VLM is good at understanding what is on a screen and roughly where things are, but bad at precise pixel coordinates. By asking for percentages instead of pixels, the estimation is more natural for the VLM. By drawing the predicted box and showing it back, the hard estimation problem becomes an easier verification problem. The grid calibration handles all the precise geometry with simple math, keeping the VLM out of coordinate calculations entirely. The retry loop at both the bounding box level and the tap result level makes the system self-correcting.

---

## File Structure

- calibration/ — Dot display HTML page, calibration script, saved calibration data and dot photo.
- core/ — Coordinate mapper, camera capture, GRBL controller, VLM client.
- prompts/ — Text files containing the three prompt templates (bbox request, bbox verify, tap verify).
- main.py — The runtime loop that ties everything together.
